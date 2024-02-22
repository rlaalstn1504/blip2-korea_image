# Python 기본 라이브러리
import os
import time
import gc
from collections import OrderedDict 

# PyTorch 및 딥러닝 관련 라이브러리
import torch 
from torch import nn
from transformers import T5Model, T5Tokenizer, T5TokenizerFast, T5ForConditionalGeneration
from timm.models.eva import Eva, eva_giant_patch14_560

# 기타 유틸리티 및 사용자 정의 모듈
from tqdm import tqdm
from data_utils import Korean_gqa, get_loader
from lavis.models import load_model_and_preprocess
from lavis.models.blip2_models.blip2_t5 import Blip2T5
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel 
from lavis.models.blip2_models.blip2_qformer import Blip2Qformer 

class CustomEvaModel(Eva):
    def __init__(self, *args, **kwargs):
        super(CustomEvaModel, self).__init__(*args, **kwargs)
        # 기존의 마지막 레이어를 Identity로 변경합니다.
        self.fc_norm = nn.Identity()
        self.head_drop = nn.Identity()
        self.head = nn.Identity() 
        
    def get_num_layer(self, var_name=""):
        if var_name in ("cls_token", "mask_token", "pos_embed"):
            return 0
        elif var_name.startswith("patch_embed"):
            return 0
        elif var_name.startswith("rel_pos_bias"):
            return len(self.blocks) - 1
        elif var_name.startswith("blocks"):
            layer_id = int(var_name.split('.')[1])
            return layer_id + 1
        else:
            return len(self.blocks)
        
    def forward_features(self, x):
        x = self.patch_embed(x)
        x, rot_pos_embed = self._pos_embed(x)
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, rope=rot_pos_embed)
            else:
                x = blk(x, rope=rot_pos_embed)
        # x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        #x = self.forward_head(x)
        return x

# pretrain 1에서 사용하는 모델
class CustomBlip2_prtrain1_model(Blip2Qformer):
    def __init__(self, *args, **kwargs):  
        super(CustomBlip2_prtrain1_model, self).__init__(*args, **kwargs)
        vit_model="eva_clip_g"
        img_size=560
        drop_path_rate=0
        use_grad_checkpoint=False
        vit_precision="fp16"
        freeze_vit=True
        cross_attention_freq=2
        embed_dim=256
        max_txt_len=32  
        
        self.num_query_token = 32 
        #self.tokenizer = T5Tokenizer.from_pretrained('digit82/kolang-t5-base')
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        
        self.visual_encoder = CustomEvaModel(img_size=560, patch_size=14, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=6144 / 1408) 
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False  
            
        self.Qformer, self.query_tokens = self.init_Qformer(
            self.num_query_token, self.visual_encoder.num_features, cross_attention_freq
        ) 
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
        #print('vision_width:', vision_width)
        encoder_config = BertConfig.from_pretrained("bert-base-multilingual-cased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel.from_pretrained(
            "bert-base-multilingual-cased", config=encoder_config
        )
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens


# pretrain 2, fine-tuning에서 사용하는 모델
class CustomBlip2T5model(Blip2T5):
    def __init__(self, *args, **kwargs):   
        
        
        super(CustomBlip2T5model, self).__init__(*args, **kwargs)
        # 한국어 T5모델과 토크나이저로 기존 모델의 구성을 대체해 줍니다.         
        self.tokenizer = T5TokenizerFast.from_pretrained('paust/pko-chat-t5-large')
        self.t5_tokenizer = T5TokenizerFast.from_pretrained('paust/pko-chat-t5-large')
        self.t5_model = T5ForConditionalGeneration.from_pretrained('paust/pko-chat-t5-large')
        self.visual_encoder = CustomEvaModel(img_size=560, patch_size=14, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=6144 / 1408)
        
        # eva 모델의 pretrain 가중치를 불러옴
        eva_pretrain_model = eva_giant_patch14_560(pretrained=True)  
        eva_pretrain_model.fc_norm = nn.Identity() 
        eva_pretrain_model.head_drop = nn.Identity()
        eva_pretrain_model.head = nn.Identity() 

        # blip2 내부의 eva 모델에 pretrain 가중치 삽입
        self.visual_encoder.load_state_dict(eva_pretrain_model.state_dict())  
        del eva_pretrain_model # eva 모델 삭제
        
        # ViT 모델과 LLM 모델의 가중치 freeze
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False 
            
        self.num_query_token = 32 
        self.max_txt_len = 32
        self.Qformer, self.query_tokens = self.init_Qformer(
            self.num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None  
        for name, param in self.t5_model.named_parameters():
            param.requires_grad = False
            param.data = param.data.bfloat16()

        self.t5_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.t5_model.config.hidden_size
        )

        self.max_txt_len = 32
        self.prompt = ''

        self._apply_lemmatizer = False
        self._lemmatizer = None    
            
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
        #print('vision_width:', vision_width)
        encoder_config = BertConfig.from_pretrained("bert-base-multilingual-cased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel.from_pretrained(
            "bert-base-multilingual-cased", config=encoder_config
        )
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens
    
    def forward(self, samples):
        #print(list(self.parameters()))
        image = samples["image"]

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        
        inputs_t5 = self.t5_proj(query_output.last_hidden_state) # Qformer를 거치고 t5에 들어갈 이미지 임베딩 값 (크기 맞춰준 것)
        #inputs_t5 = self.t5_proj_connecter(inputs_t5) # shape 맞추기 위해 추가한 레이어
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)
        # 여기까지가 이미지에 대한 Qformer 변환 과정
        #with self.maybe_autocast(dtype=torch.bfloat16): 
        with self.maybe_autocast(dtype=torch.float16):
            input_tokens = self.t5_tokenizer(
                samples["text_input"],
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            output_tokens = self.t5_tokenizer(
                samples["text_output"],
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)

            encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            )

            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=output_tokens.attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss

            return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_t5 = self.t5_proj(query_output.last_hidden_state) 
        #inputs_t5 = self.t5_proj_connecter(inputs_t5) # shape 맞추기 위해 추가
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.prompt

        if isinstance(prompt, str):
            prompt = [prompt] * image.size(0)
        else:
            assert len(prompt) == image.size(
                0
            ), "The number of prompts must be equal to the batch size."

        input_tokens = self.t5_tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).to(image.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        # GPU 기종에 따라 bfloat16 안되는 것 존재 
        # with self.maybe_autocast(dtype=torch.bfloat16): 
        with self.maybe_autocast(dtype=torch.float16):
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

        return output_text

def remove_module_prefix(state_dict):
    """추론 시 멀티 GPU 모델의 state_dict에서 'module.' 접두사를 제거합니다."""
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")  # 'module.' 문자열을 제거합니다.
        new_state_dict[new_key] = value
    return new_state_dict
# 모델 생성 예시 
# pretrain1 모델 : model = CustomBlip2_prtrain1_model()
# 파인튜닝 모델 : blipT5_model = CustomBlip2T5model(img_size=560, num_query_token = 32)
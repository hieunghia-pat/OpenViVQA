import functools

import torch
from torch import nn
import torch.nn.functional as F
from torch import nn
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import *
from transformers.generation.utils import *

from utils.logging_utils import Logger
from utils.instance import InstanceList
from builders.model_builder import META_ARCHITECTURE
from models.modules.TSS import TextSemanticSeparate
from models.modules.SCP import SpatialCirclePosition

logger = Logger()

@META_ARCHITECTURE.register()
class ATNet(T5ForConditionalGeneration):
    def __init__(self, config, vocab):
        t5_config = T5Config(
            vocab_size=len(vocab),
            d_model=config.D_MODEL,
            num_layers=config.MMT.NUM_HIDDEN_LAYERS,
            num_heads=config.MMT.NUM_ATTENTION_HEADS,
            num_decoder_layers=config.MMT.NUM_HIDDEN_LAYERS,
            d_ff=config.MMT.D_FF,
            d_kv=config.MMT.D_KV,
            bos_token_id=vocab.bos_idx,
            pad_token_id=vocab.padding_idx,
            unk_token_id=vocab.unk_idx,
            eos_token_id=vocab.eos_idx
        )

        super().__init__(t5_config)

        self.vocab = vocab
        self.max_iter = vocab.max_answer_length
        self.object_dim = config.OBJECT_EMBEDDING.D_FEATURE
        self.ocr_dim = config.OCR_EMBEDDING.D_FEATURE
        self.d_model = self.config.hidden_size
        self.tss_config = config.TSS
        self.scp_config = config.SCP

        self.build()

    def build(self):
        self._build_obj_encoding()
        self._build_ocr_encoding()

    def _build_obj_encoding(self):
        self.linear_obj_feat_to_mmt_in = nn.Linear(
            self.object_dim, 
            self.config.hidden_size
        )

        # object location feature: relative bounding box coordinates (4-dim)
        self.linear_obj_bbox_to_mmt_in = nn.Linear(4, self.config.hidden_size)

        self.obj_feat_layer_norm = T5LayerNorm(self.config.hidden_size)
        self.obj_bbox_layer_norm = T5LayerNorm(self.config.hidden_size)
        self.obj_drop = nn.Dropout(self.config.dropout_rate)

    def _build_ocr_encoding(self):
        self.linear_ocr_feat_to_mmt_in = nn.Linear(
            self.ocr_dim, 
            self.config.hidden_size
        )

        # OCR location feature: relative bounding box coordinates (4-dim)
        self.linear_ocr_bbox_to_mmt_in = nn.Linear(4, self.config.hidden_size)

        self.ocr_feat_layer_norm = T5LayerNorm(self.config.hidden_size)
        self.ocr_bbox_layer_norm = T5LayerNorm(self.config.hidden_size)
        self.ocr_text_layer_norm = T5LayerNorm(self.config.hidden_size)
        self.ocr_drop = nn.Dropout(self.config.dropout_rate)

        self.tss = TextSemanticSeparate(self.tss_config)
        self.tss_layer_norm = T5LayerNorm(self.config.hidden_size)

        self.scp = SpatialCirclePosition(self.scp_config)
        self.scp_layer_norm = T5LayerNorm(self.config.hidden_size)

    def forward(self, items, **kwargs):
        fwd_results = {}
        self._foward_embedding(items, fwd_results)

        if self.training:
            answer_tokens = items.answer_tokens
            shifted_right_answer_tokens = items.shifted_right_answer_tokens
            bs, seq_len = answer_tokens.shape
            decoder_attention_mask = _get_causal_mask(bs, seq_len, self.device)
        
            return super().forward(
                inputs_embeds=fwd_results["mmt_in"],
                attention_mask=fwd_results["mmt_mask"],
                decoder_input_ids=answer_tokens,
                decoder_attention_mask=decoder_attention_mask,
                labels=shifted_right_answer_tokens
            )

        bs = fwd_results["mmt_in"].shape[0]
        kwargs["encoder_outputs"] = None
        kwargs["attention_mask"] = fwd_results["mmt_mask"]
        kwargs["decoder_attention_mask"] = _get_causal_mask(bs, 1, self.device)
        return super().forward(
                inputs_embeds=fwd_results["mmt_in"],
                **kwargs
            )
        
    def _forward_txt_embedding(self, items, fwd_results):
        fwd_results["txt_inds"] = items.question_tokens

        # binary mask of valid text (question words) vs padding
        text_len = (items.question_tokens != self.vocab.padding_idx).sum(dim=-1)
        fwd_results["txt_mask"] = _get_mask(
            text_len, items.question_tokens.size(1)
        )
        fwd_results["txt_emb"] = self.shared(fwd_results["txt_inds"])

    def _forward_obj_embedding(self, items, fwd_results):
        # object appearance feature
        obj_feat = items.region_features
        obj_bbox = items.region_boxes
        obj_tag = items.object_list.long()

        obj_mmt_in = self.obj_feat_layer_norm(
            self.linear_obj_feat_to_mmt_in(obj_feat)
        ) + self.obj_bbox_layer_norm(
            self.linear_obj_bbox_to_mmt_in(obj_bbox)
        ) + self.shared(obj_tag)
        obj_mmt_in = self.obj_drop(obj_mmt_in)
        fwd_results["obj_mmt_in"] = obj_mmt_in

        # binary mask of valid object vs padding
        obj_nums = (items.region_features.sum(dim=-1) != 0).sum(dim=-1)
        fwd_results["obj_mask"] = _get_mask(obj_nums, obj_mmt_in.size(1))

    def _forward_ocr_embedding(self, items, fwd_results):
        # OCR rec feature (256-dim), replace the OCR PHOC features, extracted from swintextspotter
        ocr_phoc = items.ocr_rec_features
        ocr_phoc = F.normalize(ocr_phoc, dim=-1)

        # OCR appearance feature, extracted from swintextspotter
        ocr_fc = items.ocr_det_features
        ocr_fc = F.normalize(ocr_fc, dim=-1)

        ocr_feat = torch.cat(
            [ocr_phoc, ocr_fc], dim=-1
        )
        ocr_emb = self.ocr_feat_layer_norm(
            self.linear_ocr_feat_to_mmt_in(ocr_feat)
        )

        ocr_bbox = items.ocr_boxes
        ocr_box_emb = self.ocr_bbox_layer_norm(
            self.linear_ocr_bbox_to_mmt_in(ocr_bbox)
        )
        
        ocr_text = items.ocr_tokens.long()
        ocr_text_emb = F.normalize(self.shared(ocr_text), dim=-1)

        # binary mask of valid OCR vs padding
        ocr_nums = (items.ocr_det_features.sum(dim=-1) != 0).sum(dim=-1)
        ocr_mask = _get_mask(ocr_nums, ocr_emb.size(1))

        ocr_tss = self.tss_layer_norm(self.tss(
            ocr_emb,
            ocr_box_emb,
            ocr_text_emb
        ))

        ocr_scp = self.scp_layer_norm(self.scp(
            ocr_features=ocr_tss,
            ocr_boxes=ocr_bbox, 
            ocr_padding_masks=ocr_mask, 
            image_sizes=items.image_size
        ))

        fwd_results["ocr_mmt_in"] = ocr_scp
        fwd_results["ocr_mask"] = ocr_mask

    def _foward_embedding(self, items, fwd_results):
        self._forward_txt_embedding(items, fwd_results)
        self._forward_obj_embedding(items, fwd_results)
        self._forward_ocr_embedding(items, fwd_results)
        fwd_results["mmt_in"] = torch.cat([
            fwd_results["txt_emb"], 
            fwd_results["obj_mmt_in"], 
            fwd_results["ocr_mmt_in"]], dim=1)
        fwd_results["mmt_mask"] = torch.cat([
            fwd_results["txt_mask"],
            fwd_results["obj_mask"],
            fwd_results["ocr_mask"]
        ], dim=-1)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        items,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "items": items,
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }
    
    @torch.no_grad()
    def generate(self, items, **kwargs) -> torch.LongTensor:
        fwd_results = {}
        self._foward_embedding(items, fwd_results)

        inputs = None
        inputs_embeds = fwd_results["mmt_in"]
        generation_config = None
        logits_processor = None
        stopping_criteria = None
        prefix_allowed_tokens_fn = None
        synced_gpus: Optional[bool] = None
        negative_prompt_ids: Optional[torch.Tensor] = None
        negative_prompt_attention_mask: Optional[torch.Tensor] = None

        if synced_gpus is None:
            if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
                synced_gpus = True
            else:
                synced_gpus = False

        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self._validate_model_class()

        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        if generation_config is None:
            # legacy: users may modify the model configuration to control generation. To trigger this legacy behavior,
            # two conditions must be met
            # 1) the generation config must have been created from the model config (`_from_model_config` field);
            # 2) the generation config must have seen no modification since its creation (the hash is the same).
            if self.generation_config._from_model_config and self.generation_config._original_object_hash == hash(
                self.generation_config
            ):
                new_generation_config = GenerationConfig.from_model_config(self.config)
                if new_generation_config != self.generation_config:
                    warnings.warn(
                        "You have modified the pretrained model configuration to control generation. This is a"
                        " deprecated strategy to control generation and will be removed soon, in a future version."
                        " Please use and modify the model generation configuration (see"
                        " https://huggingface.co/docs/transformers/generation_strategies#default-text-generation-configuration )"
                    )
                    self.generation_config = new_generation_config
            generation_config = self.generation_config

        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
        generation_config.validate()
        self._validate_model_kwargs(model_kwargs.copy())

        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            if model_kwargs.get("attention_mask", None) is None:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            generation_config.pad_token_id = eos_token_id

        # 3. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        inputs_tensor = inputs_tensor.repeat(items.batch_size, 1)
        batch_size = inputs_tensor.shape[0]

        # 4. Define other model kwargs
        model_kwargs["output_attentions"] = generation_config.output_attentions
        model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            model_kwargs["use_cache"] = True
        else:
            model_kwargs["use_cache"] = generation_config.use_cache

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
            )

        # decoder-only models should use left-padding for generation
        if not self.config.is_encoder_decoder:
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                generation_config.pad_token_id is not None
                and len(inputs_tensor.shape) == 2
                and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created
            # and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config.decoder_start_token_id,
                bos_token_id=generation_config.bos_token_id,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        generation_config.max_length = self.vocab.max_answer_length
        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        if self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 8. prepare distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

        # 9. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )

        return self.greedy_search(
                input_ids,
                items,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )
    
    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        items: InstanceList,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, torch.LongTensor]:
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, items, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores):
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GreedySearchEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return GreedySearchDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids

def _get_mask(nums, max_num):
    # non_pad_mask: b x lq, torch.float32, 0. on PAD
    batch_size = nums.size(0)
    arange = torch.arange(0, max_num).unsqueeze(0).expand(batch_size, -1)
    non_pad_mask = arange.to(nums.device).lt(nums.unsqueeze(-1))
    non_pad_mask = non_pad_mask.type(torch.float32)
    return non_pad_mask

@functools.lru_cache(maxsize=32)
def _get_causal_mask(bs, seq_length, device):
    # generate a lower triangular mask
    mask = torch.zeros(bs, seq_length, seq_length, device=device)
    for i in range(seq_length):
        for j in range(i + 1):
            mask[:, i, j] = 1.0
    return mask

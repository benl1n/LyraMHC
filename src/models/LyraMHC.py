import torch
import torch.nn as nn
from omegaconf import DictConfig
from src.models.components.fusion import LiteBiCrossAttentionFusion
from src.models.components.predictors import SequencePredictor
from src.models.components.preprocess import SequenceEncoder
from src.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("LyraMHC")
class LyraMHC(nn.Module):
    def __init__(self, cfg: DictConfig):

        super(LyraMHC, self).__init__()

        self.cfg = cfg

        d_input = cfg.model.params.d_input
        d_model = cfg.model.params.d_model
        dropout = cfg.model.params.dropout
        fusion_cfg = cfg.model.fusion
        predictor_cfg = cfg.model.predictor

        self.hla_encoder = SequenceEncoder(d_input, d_model, dropout, cfg.model.encoder_hla)
        self.pep_encoder = SequenceEncoder(d_input, d_model, dropout, cfg.model.encoder_pep)

        if cfg.train.task == "TCR":
            self.tcr_encoder = SequenceEncoder(d_input, d_model, dropout, cfg.model.encoder_tcr)
            self.fusion_stack_tcr = nn.ModuleList([
                LiteBiCrossAttentionFusion(**fusion_cfg)
            ])
        self.fusion_stack = nn.ModuleList([
            LiteBiCrossAttentionFusion(**fusion_cfg)
        ])

        final_dim = d_model * 2
        pred_input_size = predictor_cfg.get('input_size', final_dim)
        self.predictor = SequencePredictor(input_size=pred_input_size)

    def forward(self,
                hla_a_seqs: torch.Tensor,
                peptides: torch.Tensor,
                tcr: torch.Tensor = None,
                return_features: bool = False):

        cfg = self.cfg

        hla_out = self.hla_encoder(hla_a_seqs)
        pep_out = self.pep_encoder(peptides)

        for fusion in self.fusion_stack:
            out = fusion(hla_out, pep_out)

        if cfg.train.task == "TCR":
            tcr_out = self.tcr_encoder(tcr)
            for fusion in self.fusion_stack_tcr:
                out = fusion(tcr_out, out)

        out_mean = torch.mean(out, dim=1)
        out_max, _ = torch.max(out, dim=1)
        all_out = torch.cat([out_mean, out_max], dim=-1)

        if return_features:
            features = {
                "hla_origin": hla_a_seqs,
                "pep_origin": peptides,
                "hla_encoded": hla_out,
                "pep_encoded": pep_out,
                "fused_interaction": all_out
            }
            # if tcr_out is not None:
            #     features["tcr_encoded"] = tcr_out

            return features

        ic50 = self.predictor(all_out)
        return ic50

    def load_pretrained_weights(self, ckpt_path, freeze=True):
        print(f"load weight: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        keep_keys = [
            "hla_encoder",
            "pep_encoder",
            "fusion_stack_pMHC"
        ]

        filtered_state = {
            k: v for k, v in state_dict.items()
            if any(k.startswith(prefix) for prefix in keep_keys)
        }

        missing, unexpected = self.load_state_dict(filtered_state, strict=False)

        if missing:
            print(f": {missing[:5]} ...")
        if unexpected:
            print(f"un para: {unexpected[:5]} ...")

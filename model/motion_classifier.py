import torch.nn as nn
import torch
from model.modiffae import InputProcess, PositionalEncoding


class MotionClassifier(nn.Module):
    def __init__(self, latent_dim, num_heads, num_layers, transformer_feedforward_dim, dropout,
                 attribute_dim, pose_rep, input_feats, num_frames, semantic_pool_type):
        super(MotionClassifier, self).__init__()

        self.semantic_pool_type = semantic_pool_type

        self.input_process = InputProcess(pose_rep, input_feats, latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(latent_dim, dropout)

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                             nhead=num_heads,
                                                             dim_feedforward=transformer_feedforward_dim,
                                                             dropout=dropout,
                                                             activation="gelu")

        self.seqTransEncoder = nn.TransformerEncoder(seq_trans_encoder_layer,
                                                     num_layers=num_layers)

        if self.semantic_pool_type == 'linear_time_layer':
            self.linear_time = nn.Linear(
                in_features=num_frames,
                out_features=1
            )

        self.classification_layer = torch.nn.Linear(latent_dim, attribute_dim)

    def create_embedding(self, x):
        x = self.input_process(x)
        x_seq = self.sequence_pos_encoder(x)  # [seqlen, bs, d]
        encoder_output = self.seqTransEncoder(x_seq)  # [seqlen, bs, d]
        output = encoder_output.transpose(2, 0)  # # [semdim, bs, seqlen]

        if self.semantic_pool_type == 'global_avg_pool':
            output = torch.mean(output, dim=-1).transpose(1, 0)
        elif self.semantic_pool_type == 'global_max_pool':
            output = torch.amax(output, dim=-1).transpose(1, 0)
        elif self.semantic_pool_type == 'linear_time_layer':
            output = self.linear_time(output).squeeze().transpose(1, 0)
        elif self.semantic_pool_type == 'gated_multi_head_attention_pooling':
            # This could be interesting
            raise Exception("Not implemented.")
        else:
            raise Exception("Pool type not implemented.")

        return output

    def forward(self, x):
        output = self.create_embedding(x)
        output = self.classification_layer(output)
        return output

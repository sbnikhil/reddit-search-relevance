import torch
import torch.nn as nn
from transformers import AutoModel


class RedditRelevanceRanker(nn.Module):
    def __init__(self, transformer_model_name, extra_feature_dim, dropout=0.1):
        super(RedditRelevanceRanker, self).__init__()
        self.transformer = AutoModel.from_pretrained(transformer_model_name)
        self.fc_text = nn.Linear(768, 128)
        self.feature_normalizer = nn.LayerNorm(extra_feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_combined = nn.Linear(128 + extra_feature_dim, 64)
        self.output = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, extra_features):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_representation = outputs.last_hidden_state[:, 0, :]
        text_x = self.relu(self.fc_text(cls_representation))
        text_x = self.dropout(text_x)
        extra_features_scaled = self.feature_normalizer(extra_features)
        combined = torch.cat((text_x, extra_features_scaled), dim=1)
        x = self.relu(self.fc_combined(combined))
        x = self.dropout(x)
        return self.output(x)

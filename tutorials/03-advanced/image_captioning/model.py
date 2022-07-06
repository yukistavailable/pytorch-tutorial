import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

    def fine_tune(self, fine_tune=True):
        for p in self.resnet.parameters():
            p.requires_grad = False
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune
        for p in self.linear.parameters():
            p.requires_grad = fine_tune
        for p in self.bn.parameters():
            p.requires_grad = fine_tune


class EncoderCNNWithAttention(nn.Module):
    def __init__(self, pixel_num=16):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNNWithAttention, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-3]  # 512*28*28
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(1024, 512)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)  # 1*1024*16*16
            features = features.permute(0, 2, 3, 1)  # 1*16*16*1024
            features = self.linear(features)  # 1*16*16*512
        return features

    def fine_tune(self, fine_tune=True):
        for p in self.resnet.parameters():
            p.requires_grad = False
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune
        for p in self.linear.parameters():
            p.requires_grad = fine_tune


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        # linear layer to transform encoded image
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        # linear layer to transform decoder's output
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        # linear layer to calculate values to be softmax-ed
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(
            encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        # (batch_size, num_pixels)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (
            encoder_out *
            alpha.unsqueeze(2)).sum(
            dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderRNNWithAttention(nn.Module):
    def __init__(
            self,
            embed_size,
            hidden_size,
            vocab_size,
            num_layers,
            encoder_size,
            device,
            attention_size=126,
            max_seq_length=20,
            dropout=0.5
    ):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNNWithAttention, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(encoder_size, hidden_size, attention_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        # linear layer to find initial hidden state of LSTMCell
        self.init_h = nn.Linear(encoder_size, hidden_size)
        # linear layer to find initial cell state of LSTMCell
        self.init_c = nn.Linear(encoder_size, hidden_size)
        self.max_seg_length = max_seq_length
        self.device = device
        self.vocab_size = vocab_size
        self.encoder_size = encoder_size
        self.dropout = nn.Dropout(p=dropout)
        self.decode_step = nn.LSTMCell(
            embed_size + encoder_size,
            hidden_size,
            bias=True)  # decoding LSTMCell
        # linear layer to find initial hidden state of LSTMCell
        self.init_h = nn.Linear(encoder_size, hidden_size)
        # linear layer to find initial cell state of LSTMCell
        self.init_c = nn.Linear(encoder_size, hidden_size)
        # linear layer to create a sigmoid-activated gate
        self.f_beta = nn.Linear(hidden_size, encoder_size)
        self.sigmoid = nn.Sigmoid()

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, features, captions, lengths, encoder_size=512):
        """Decode image feature vectors and generates captions."""
        batch_size = features.size(0)

        # Flatten image
        # (batch_size, num_pixels, encoder_size)
        features = features.view(batch_size, -1, encoder_size)
        num_pixels = features.size(1)
        lengths = torch.stack([torch.Tensor([length]) for length in lengths])
        caption_lengths, sort_ind = lengths.squeeze(
            1).sort(dim=0, descending=True)
        features = features[sort_ind]

        embeddings = self.embed(captions)
        h, c = self.init_hidden_state(features)
        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(
            batch_size,
            int(max(decode_lengths)),
            self.vocab_size).to(self.device)
        alphas = torch.zeros(
            batch_size,
            int(max(lengths)),
            num_pixels).to(
            self.device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and
        # the attention weighted encoding
        for t in range(int(max(decode_lengths))):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(
                features[:batch_size_t], h[:batch_size_t])
            # gating scalar, (batch_size_t, encoder_dim)
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.linear(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, captions[sort_ind], alphas, sort_ind, decode_lengths

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []

        # Flatten image
        # (batch_size, num_pixels, encoder_size)
        batch_size = features.size(0)
        # (batch_size, num_pixels, encoder_size)
        features = features.view(batch_size, -1, self.encoder_size)
        num_pixels = features.size(1)

        h, c = self.init_hidden_state(features)

        # set <start> token
        embeddings = self.embed(torch.LongTensor([[1]]).to(self.device))
        for i in range(self.max_seg_length):
            attention_weighted_encoding, alpha = self.attention(
                features, h)
            # gating scalar, (batch_size_t, encoder_dim)
            gate = self.sigmoid(self.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding
            if i == 0:
                h, c = self.decode_step(
                    torch.cat([embeddings[:, i, :], attention_weighted_encoding], dim=1),
                    (h, c))  # (batch_size_t, decoder_dim)
            else:
                h, c = self.decode_step(
                    torch.cat([h, attention_weighted_encoding], dim=1),
                    (h, c))  # (batch_size_t, decoder_dim)
            # (batch_size_t, vocab_size)
            outputs = self.linear(self.dropout(h))

            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
        # sampled_ids: (batch_size, max_seq_length)
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids


class DecoderRNN(nn.Module):
    def __init__(
            self,
            embed_size,
            hidden_size,
            vocab_size,
            num_layers,
            max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers,
            batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        # features.unsqueeze(1)とembeddingsを結合。embeddingsの最初にfeaturesをくっつけることで、featuresを最初の単語として扱っているイメージ
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            # hiddens: (batch_size, 1, hidden_size)
            hiddens, states = self.lstm(inputs, states)
            # outputs:  (batch_size, vocab_size)
            outputs = self.linear(hiddens.squeeze(1))
            # predicted: (batch_size)
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            # inputs: (batch_size, embed_size)
            inputs = self.embed(predicted)
            # inputs: (batch_size, 1, embed_size)
            inputs = inputs.unsqueeze(1)
        # sampled_ids: (batch_size, max_seq_length)
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids

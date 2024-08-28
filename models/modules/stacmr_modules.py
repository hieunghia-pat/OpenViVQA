import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from collections import OrderedDict
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


class EncoderRNN(nn.Module):
    def __init__(self, dim_vid, dim_hidden, input_dropout_p=0.2, rnn_dropout_p=0.5,
                 n_layers=1, bidirectional=False, rnn_cell='gru'):
        """

        Args:
            hidden_dim (int): dim of hidden state of rnn
            input_dropout_p (int): dropout probability for the input sequence
            dropout_p (float): dropout probability for the output sequence
            n_layers (int): number of rnn layers
            rnn_cell (str): type of RNN cell ('LSTM'/'GRU')
        """
        super(EncoderRNN, self).__init__()
        self.dim_vid = dim_vid
        self.dim_hidden = dim_hidden
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_cell = rnn_cell

        self.vid2hid = nn.Linear(dim_vid, dim_hidden)
        self.input_dropout = nn.Dropout(input_dropout_p)

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU

        self.rnn = self.rnn_cell(dim_hidden, dim_hidden, n_layers, batch_first=True,
                                 bidirectional=bidirectional, dropout=self.rnn_dropout_p)

        self._init_hidden()

    def _init_hidden(self):
        nn.init.xavier_normal_(self.vid2hid.weight)

    def forward(self, vid_feats):
        """
        Applies a multi-layer RNN to an input sequence.
        Args:
            input_var (batch, seq_len): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch
        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
        """
        batch_size, seq_len, dim_vid = vid_feats.size()
        vid_feats = self.vid2hid(vid_feats.reshape(-1, dim_vid))
        vid_feats = self.input_dropout(vid_feats)
        vid_feats = vid_feats.view(batch_size, seq_len, self.dim_hidden)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(vid_feats)
        return output, hidden
    

class DecoderRNN(nn.Module):
    """
    Provides functionality for decoding in a seq2seq framework, with an option for attention.
    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        dim_hidden (int): the number of features in the hidden state `h`
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        bidirectional (bool, optional): if the encoder is bidirectional (default False)
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        rnn_dropout_p (float, optional): dropout probability for the output sequence (default: 0)

    """

    def __init__(self,
                 vocab_size,
                 max_len,
                 dim_hidden,
                 dim_word,
                 n_layers=1,
                 rnn_cell='gru',
                 bidirectional=False,
                 input_dropout_p=0.1,
                 rnn_dropout_p=0.1):
        super(DecoderRNN, self).__init__()

        self.bidirectional_encoder = bidirectional
        self.dim_output = vocab_size
        self.dim_hidden = dim_hidden * 2 if bidirectional else dim_hidden
        self.dim_word = dim_word
        self.max_length = max_len
        self.sos_id = 1
        self.eos_id = 2
        self.input_dropout = nn.Dropout(input_dropout_p)
        self.embedding = nn.Embedding(17000, dim_word)
        self.attention = Attention(self.dim_hidden)
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        self.rnn = self.rnn_cell(
            self.dim_hidden + dim_word,
            self.dim_hidden,
            n_layers,
            batch_first=True,
            dropout=rnn_dropout_p)

        self.out = nn.Linear(self.dim_hidden, self.dim_output)

        self._init_weights()

    def forward(self,
                encoder_outputs,
                encoder_hidden,
                targets=None,
                mode='train',
                opt={}):
        """

        Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        - **encoder_hidden** (num_layers * num_directions, batch_size, dim_hidden): tensor containing the features in the
          hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
        - **encoder_outputs** (batch, seq_len, dim_hidden * num_directions): (default is `None`).
        - **targets** (batch, max_length): targets labels of the ground truth sentences

        Outputs: seq_probs,
        - **seq_logprobs** (batch_size, max_length, vocab_size): tensors containing the outputs of the decoding function.
        - **seq_preds** (batch_size, max_length): predicted symbols
        """
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)

        batch_size, _, _ = encoder_outputs.size()
        decoder_hidden = self._init_rnn_state(encoder_hidden)

        seq_logprobs = []
        seq_preds = []
        self.rnn.flatten_parameters()

        if mode == 'train':
            # use targets as rnn inputs
            # print(targets)
            targets_emb = self.embedding(targets)

            for i in range(self.max_length - 1):
                current_words = targets_emb[:, i, :]
                context = self.attention(decoder_hidden.squeeze(0), encoder_outputs)
                decoder_input = torch.cat([current_words, context], dim=1)
                decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
                decoder_output, decoder_hidden = self.rnn(
                    decoder_input, decoder_hidden)
                logprobs = F.log_softmax(
                    self.out(decoder_output.squeeze(1)), dim=1)
                seq_logprobs.append(logprobs.unsqueeze(1))

            seq_logprobs = torch.cat(seq_logprobs, 1)

        elif mode == 'inference':
            if beam_size > 1:
                return self.sample_beam(encoder_outputs, decoder_hidden, opt)

            for t in range(self.max_length - 1):
                context = self.attention(
                    decoder_hidden.squeeze(0), encoder_outputs)

                if t == 0:  # input <bos>
                    it = torch.LongTensor([self.sos_id] * batch_size).cuda()
                elif sample_max:
                    sampleLogprobs, it = torch.max(logprobs, 1)
                    # seq_logprobs.append(sampleLogprobs.view(-1, 1))
                    seq_logprobs.append(logprobs.unsqueeze(dim=1))
                    it = it.view(-1).long()

                else:
                    # sample according to distribuition
                    if temperature == 1.0:
                        prob_prev = torch.exp(logprobs)
                    else:
                        # scale logprobs by temperature
                        prob_prev = torch.exp(torch.div(logprobs, temperature))
                    it = torch.multinomial(prob_prev, 1).cuda()
                    sampleLogprobs = logprobs.gather(1, it)
                    # seq_logprobs.append(sampleLogprobs.view(-1, 1))
                    seq_logprobs.append(logprobs.unsqueeze(dim=1))
                    it = it.view(-1).long()

                seq_preds.append(it.view(-1, 1))

                xt = self.embedding(it)
                decoder_input = torch.cat([xt, context], dim=1)
                decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
                decoder_output, decoder_hidden = self.rnn(
                    decoder_input, decoder_hidden)
                logprobs = F.log_softmax(
                    self.out(decoder_output.squeeze(1)), dim=1)

            seq_logprobs = torch.cat(seq_logprobs, 1)
            seq_preds = torch.cat(seq_preds[1:], 1)

        return seq_logprobs, seq_preds

    def _init_weights(self):
        """ init the weight of some layers
        """
        nn.init.xavier_normal_(self.out.weight)

    def _init_rnn_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple(
                [self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, dim_hidden) -> (#layers, #batch, #directions * dim_hidden)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h


class Attention(nn.Module):
    """
    Applies an attention mechanism on the output features from the decoder.
    """

    def __init__(self, dim):
        super(Attention, self).__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim * 2, dim)
        self.linear2 = nn.Linear(dim, 1, bias=False)
        self.tanh = nn.Tanh()
        #self._init_hidden()

    def _init_hidden(self):
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)

    def forward(self, hidden_state, encoder_outputs):
        """
        Arguments:
            hidden_state {Variable} -- batch_size x dim
            encoder_outputs {Variable} -- batch_size x seq_len x dim

        Returns:
            Variable -- context vector of size batch_size x dim
        """
        batch_size, seq_len, _ = encoder_outputs.size()
        hidden_state = hidden_state.unsqueeze(1).repeat(1, seq_len, 1)
        inputs = torch.cat((encoder_outputs, hidden_state),
                           2).view(-1, self.dim * 2)
        o = self.linear2(self.tanh(self.linear1(inputs)))
        e = o.view(batch_size, seq_len)
        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)
        return context


class Rs_GCN(nn.Module):

    def __init__(self, in_channels, inter_channels, bn_layer=True):
        super(Rs_GCN, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1


        conv_nd = nn.Conv1d
        max_pool = nn.MaxPool1d
        bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant(self.W[1].weight, 0)
            nn.init.constant(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant(self.W.weight, 0)
            nn.init.constant(self.W.bias, 0)

        self.theta = None
        self.phi = None


        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)


    def forward(self, v):
        '''
        :param v: (B, D, N)
        :return:
        '''
        batch_size = v.size(0)

        g_v = self.g(v).view(batch_size, self.inter_channels, -1)
        g_v = g_v.permute(0, 2, 1)

        theta_v = self.theta(v).view(batch_size, self.inter_channels, -1)
        theta_v = theta_v.permute(0, 2, 1)
        phi_v = self.phi(v).view(batch_size, self.inter_channels, -1)
        R = torch.matmul(theta_v, phi_v)
        N = R.size(-1)
        R_div_C = R / N

        y = torch.matmul(R_div_C, g_v)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *v.size()[2:])
        W_y = self.W(y)
        v_star = W_y + v

        return v_star
    

class ObjectEncoder(nn.Module):
  def __init__(self, obj_in_dim, hidden_size, dropout_prob=0.1):
    super().__init__()

    # 2048 (FasterRCNN)
    self.linear_obj_feat_to_mmt_in = nn.Linear(obj_in_dim, hidden_size)

    # OBJ location feature
    self.linear_obj_bbox_to_mmt_in = nn.Linear(4, hidden_size)

    self.obj_feat_layer_norm = nn.LayerNorm(hidden_size)
    self.obj_bbox_layer_norm = nn.LayerNorm(hidden_size)
    self.dropout = nn.Dropout(dropout_prob)

  def forward(self, obj_boxes, obj_features):

    # Features to hidden size
    obj_features = F.normalize(obj_features, dim=-1)

    # Get obj features
    obj_features = self.obj_feat_layer_norm(self.linear_obj_feat_to_mmt_in(obj_features))
    obj_bbox_features = self.obj_bbox_layer_norm(self.linear_obj_bbox_to_mmt_in(obj_boxes))

    return self.dropout(obj_features + obj_bbox_features) # batch_size, seq_length, hidden_size


class OCREncoder(nn.Module):
    
  def __init__(self, ocr_in_dim, hidden_size, dropout_prob=0.1):
    super().__init__()

    # 300 (FastText) + 256 (rec_features) + 256 (det_features) = 812
    self.linear_ocr_feat_to_mmt_in = nn.Linear(ocr_in_dim, hidden_size)

    # OCR location feature
    self.linear_ocr_bbox_to_mmt_in = nn.Linear(4, hidden_size)

    self.ocr_feat_layer_norm = nn.LayerNorm(hidden_size)
    self.ocr_bbox_layer_norm = nn.LayerNorm(hidden_size)
    self.dropout = nn.Dropout(dropout_prob)

  def forward(self, ocr_boxes, ocr_token_embeddings, ocr_rec_features, ocr_det_features):

    # Normalize input
    ocr_token_embeddings = F.normalize(ocr_token_embeddings, dim=-1)
    ocr_rec_features = F.normalize(ocr_rec_features, dim=-1)
    ocr_det_features = F.normalize(ocr_det_features, dim=-1)

    max_len = max(ocr_token_embeddings.shape[1], ocr_rec_features.shape[1], ocr_det_features.shape[1])
    
    if ocr_token_embeddings.shape[1] < max_len:
        ocr_token_embeddings = torch.nn.functional.pad(ocr_token_embeddings,
                                                       (0, 0, 0, max_len - ocr_token_embeddings.shape[1], 0, 0))
    if ocr_rec_features.shape[1] < max_len:
        ocr_rec_features = torch.nn.functional.pad(ocr_rec_features,
                                                   (0, 0, 0, max_len - ocr_rec_features.shape[1], 0, 0))
    if ocr_det_features.shape[1] < max_len:
        ocr_det_features = torch.nn.functional.pad(ocr_det_features,
                                                   (0, 0, 0, max_len - ocr_det_features.shape[1], 0, 0))
        
    # get OCR combine features
    ocr_combine_features = torch.cat(
        [ocr_token_embeddings, ocr_rec_features, ocr_det_features], dim=-1)
    ocr_combine_features = self.ocr_feat_layer_norm(self.linear_ocr_feat_to_mmt_in(ocr_combine_features))

    # Get OCR bbox features
    ocr_bbox_features = self.ocr_bbox_layer_norm(self.linear_ocr_bbox_to_mmt_in(ocr_boxes))

    return self.dropout(ocr_combine_features + ocr_bbox_features) # batch_size, seq_length, hidden_size



class EncoderImagePrecompAttn(nn.Module):

    def __init__(self, embed_size, use_abs=False, no_imgnorm=False):
        super(EncoderImagePrecompAttn, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs

        self.bn = nn.BatchNorm1d(embed_size)

        # GSR
        self.img_rnn = nn.GRU(embed_size, embed_size, 1, batch_first=True)

        # GCN reasoning
        self.Rs_GCN_1 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.Rs_GCN_2 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.Rs_GCN_3 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.Rs_GCN_4 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)

        self.bn = nn.BatchNorm1d(embed_size)


        # GCN reasoning
        self.Text_GCN_1 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.Text_GCN_2 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.Text_GCN_3 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.Text_GCN_4 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)


    def forward(self, images, scene_text, train=True):
        """Extract image feature vectors."""

        # IMAGE FEATURES
        # fc_img_emd = self.fc(images)
        fc_img_emd = l2norm(images)


        # fc_img_emd = torch.cat((fc_img_emd, fc_scene_text), dim=1)

        # GCN reasoning
        # -> B,D,N
        GCN_img_emd = fc_img_emd.permute(0, 2, 1)
        GCN_img_emd = self.Rs_GCN_1(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_2(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_3(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_4(GCN_img_emd)
        # -> B,N,D
        GCN_img_emd = GCN_img_emd.permute(0, 2, 1)
        GCN_img_emd = l2norm(GCN_img_emd)

        rnn_img, hidden_state = self.img_rnn(GCN_img_emd)
        visual_features = hidden_state[0]

        # SCENE TEXT FEATURES --- AM --------

        fc_scene_text = F.leaky_relu(scene_text)
        fc_scene_text = l2norm(fc_scene_text)

        # Scene Text Reasoning
        # -> B,D,N
        GCN_scene_text_emd = fc_scene_text.permute(0, 2, 1)
        GCN_scene_text_emd = self.Text_GCN_1(GCN_scene_text_emd)
        GCN_scene_text_emd = self.Text_GCN_2(GCN_scene_text_emd)
        GCN_scene_text_emd = self.Text_GCN_3(GCN_scene_text_emd)
        GCN_scene_text_emd = self.Text_GCN_4(GCN_scene_text_emd)
        # # -> B,N,D
        GCN_scene_text_emd = GCN_scene_text_emd.permute(0, 2, 1)
        GCN_scene_text_emd = l2norm(GCN_scene_text_emd)
        fc_scene_text = torch.mean(GCN_scene_text_emd, dim=1)

        # FINAL AGGREGATION
        # fc_scene_text = torch.mean(fc_scene_text, dim=1)
        features = torch.mul(visual_features, fc_scene_text) + visual_features

        # features = torch.mean(GCN_img_emd, dim=1)

        features = self.bn(features)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # take the absolute value of embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features, GCN_img_emd

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecompAttn, self).load_state_dict(new_state)
        

# tutorials/08 - Language Model
# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_abs=False):
        super(EncoderText, self).__init__()
        self.use_abs = use_abs
        self.embed_size = embed_size

        # word embedding
        self.embed = nn.Embedding(17000, word_dim)

        # caption embedding
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True)

        self.init_weights()


    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)

        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)


        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        I = torch.LongTensor(lengths).view(-1, 1, 1)
        I = Variable(I.expand(x.size(0), 1, self.embed_size)-1).cuda()
        torch.cuda.synchronize()
        out = torch.gather(padded[0], 1, I).squeeze(1)

        # normalization in the joint embedding space
        out = l2norm(out)

        # take absolute value, used by order embeddings
        if self.use_abs:
            out = torch.abs(out)

        return out
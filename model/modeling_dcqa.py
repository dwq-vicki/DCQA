#coding=utf-8
from transformers import T5ForConditionalGeneration
from transformers.file_utils import ModelOutput
import torch
from .attention import ChoiceAttention, Crossattention
from torch import nn

device = torch.device("cuda:0")
class DCQA(nn.Module):
    def __init__(self, model_path, num_hidden_layers, alpha, beta):
        super(DCQA, self).__init__()
        self.alpha = alpha
        self.beta = beta

        device = torch.device("cuda:0")
        print("GPU Device:【{}：{}】".format(device.type, device.index))
        self.t5_model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
        dim = self.t5_model.config.d_model
        self.option_linear = nn.Linear(dim*2, 1).to(device)
        self.criterion = nn.CrossEntropyLoss()

        # Compute the choice attention
        self.option_att = ChoiceAttention(dim).to(device)

        # Compute the contextual information in the question related to commonalities
        self.s2q_att = Crossattention(dim, num_hidden_layers).to(device)

        # Compute the contextual information in the question related to each choice
        self.o2q_att = Crossattention(dim, num_hidden_layers).to(device)

        # Enhance the representation of each choice
        self.q2a_att = Crossattention(dim, num_hidden_layers).to(device)


        '''
        n_gpu = torch.cuda.device_count()
        print(n_gpu)
        layer_num = self.t5_model.config.num_layers
        layer_per_gpu = layer_num // n_gpu
        device_map = {}
        for n in range(n_gpu):
            device_map[n] = [i + n * layer_per_gpu for i in range(layer_per_gpu)]
        remain_layer = [i + n_gpu * layer_per_gpu for i in range(layer_num - layer_per_gpu * n_gpu)]
        device_map[n_gpu - 1] += remain_layer
        self.t5_model.parallelize(device_map)
        '''

    def forward(self, q_ids, q_mask, qo_ids, qo_mask, choice_num, clue_ids=None, answers=None):
        self.choice_num = choice_num
        if answers is not None and clue_ids is not None:
            opt_score, output_sequences, sq_weight, oq_weight = self.get_option_score(q_ids, q_mask, qo_ids, qo_mask)
            local_device = self.t5_model.device
            t5_output = self.t5_model(input_ids=q_ids.to(local_device), attention_mask=q_mask.to(local_device),
                                      labels=clue_ids.to(local_device))
            loss_ans = t5_output.loss
            loss = self.criterion(opt_score, answers)
            return self.alpha * loss + self.beta * loss_ans
        else:
            opt_score, output_sequences, sq_weight, oq_weight = self.get_option_score(q_ids, q_mask, qo_ids, qo_mask)
            return opt_score, output_sequences, sq_weight, oq_weight

    def get_option_score(self, q_ids, q_mask, qo_ids, qo_mask):
        local_device = self.t5_model.encoder.device
        t5_output = self.t5_model.encoder(input_ids=qo_ids.to(local_device), attention_mask=qo_mask.to(local_device))
        encoder_qo = t5_output[0]

        t5_output = self.t5_model.encoder(input_ids=q_ids.to(local_device), attention_mask=q_mask.to(local_device))
        encoder_q = t5_output[0]

        encoder_q = encoder_q.unsqueeze(dim=1)
        # print(decoder_qo.shape)#8X1X...X...
        encoder_q = encoder_q.expand(
            [encoder_q.size(0), self.choice_num, encoder_q.size(-2), encoder_q.size(-1)]).contiguous()
        # print(decoder_qo.shape)#8X5X...X...
        encoder_q = encoder_q.view(-1, encoder_q.size(-2), encoder_q.size(-1))
        # print(encoder_q.shape) # 40X...X...

        q_mask = q_mask.unsqueeze(dim=1)
        q_mask = q_mask.expand(
            [q_mask.size(0), self.choice_num, q_mask.size(-1)]).contiguous()
        q_mask = q_mask.view(-1, q_mask.size(-1))
        '''
        # obqa, arc-easy, arc-challenge
        option_vecs0 = []
        option_vecs1 = []
        option_vecs2 = []
        option_vecs3 = []
        for i in range(int(len(encoder_q) / 4)):
            option_vecs0.append(encoder_qo[i * 4, ..., ...])
            option_vecs1.append(encoder_qo[i * 4 + 1, ..., ...])
            option_vecs2.append(encoder_qo[i * 4 + 2, ..., ...])
            option_vecs3.append(encoder_qo[i * 4 + 3, ..., ...])
            
        # compute the similarity between choices
        option_vecs0 = torch.tensor([item.detach().cpu().numpy() for item in option_vecs0]).to(local_device)
        option_vecs1 = torch.tensor([item.detach().cpu().numpy() for item in option_vecs1]).to(local_device)
        option_vecs2 = torch.tensor([item.detach().cpu().numpy() for item in option_vecs2]).to(local_device)
        option_vecs3 = torch.tensor([item.detach().cpu().numpy() for item in option_vecs3]).to(local_device)

        option_similar_vecs = self.option_att(option_vecs0, option_vecs1, option_vecs2, option_vecs3)
        option_similar = []

        for i in range(len(option_similar_vecs)):
            for j in range(4):
                option_similar.append(option_similar_vecs[i])

        option_similar = torch.tensor([item.detach().cpu().numpy() for item in option_similar]).to(local_device)
        '''
        '''
        # piqa
        option_vecs0 = []
        option_vecs1 = []
        for i in range(int(len(encoder_q) / 2)):
            option_vecs0.append(encoder_qo[i * 2, ..., ...])
            option_vecs1.append(encoder_qo[i * 2 + 1, ..., ...])
        # compute the similarity between choices
        option_vecs0 = torch.tensor([item.detach().cpu().numpy() for item in option_vecs0]).to(local_device)
        option_vecs1 = torch.tensor([item.detach().cpu().numpy() for item in option_vecs1]).to(local_device)

        option_similar_vecs = self.option_att(option_vecs0, option_vecs1)
        option_similar = []

        for i in range(len(option_similar_vecs)):
            for j in range(2):
                option_similar.append(option_similar_vecs[i])

        option_similar = torch.tensor([item.detach().cpu().numpy() for item in option_similar]).to(local_device)

        '''
        '''
        # socialiqa
        option_vecs0 = []
        option_vecs1 = []
        option_vecs2 = []
        for i in range(int(len(encoder_q) / 3)):
            option_vecs0.append(encoder_qo[i * 3, ..., ...])
            option_vecs1.append(encoder_qo[i * 3 + 1, ..., ...])
            option_vecs2.append(encoder_qo[i * 3 + 2, ..., ...])
        # compute the similarity between choices
        option_vecs0 = torch.tensor([item.detach().cpu().numpy() for item in option_vecs0]).to(local_device)
        option_vecs1 = torch.tensor([item.detach().cpu().numpy() for item in option_vecs1]).to(local_device)
        option_vecs2 = torch.tensor([item.detach().cpu().numpy() for item in option_vecs2]).to(local_device)

        option_similar_vecs = self.option_att(option_vecs0, option_vecs1, option_vecs2)

        option_similar = []

        for i in range(len(option_similar_vecs)):
            for j in range(3):
                option_similar.append(option_similar_vecs[i])

        option_similar = torch.tensor([item.detach().cpu().numpy() for item in option_similar]).to(local_device)

        '''
        '''
        #qasc
        option_vecs0 = []
        option_vecs1 = []
        option_vecs2 = []
        option_vecs3 = []
        option_vecs4 = []
        option_vecs5 = []
        option_vecs6 = []
        option_vecs7 = []
        for i in range(int(len(encoder_q) / 8)):
            option_vecs0.append(encoder_qo[i * 8, ..., ...])
            option_vecs1.append(encoder_qo[i * 8 + 1, ..., ...])
            option_vecs2.append(encoder_qo[i * 8 + 2, ..., ...])
            option_vecs3.append(encoder_qo[i * 8 + 3, ..., ...])
            option_vecs4.append(encoder_qo[i * 8 + 4, ..., ...])
            option_vecs5.append(encoder_qo[i * 8 + 5, ..., ...])
            option_vecs6.append(encoder_qo[i * 8 + 6, ..., ...])
            option_vecs7.append(encoder_qo[i * 8 + 7, ..., ...])
        # compute the similarity between choices
        option_vecs0 = torch.tensor([item.detach().cpu().numpy() for item in option_vecs0]).to(local_device)
        option_vecs1 = torch.tensor([item.detach().cpu().numpy() for item in option_vecs1]).to(local_device)
        option_vecs2 = torch.tensor([item.detach().cpu().numpy() for item in option_vecs2]).to(local_device)
        option_vecs3 = torch.tensor([item.detach().cpu().numpy() for item in option_vecs3]).to(local_device)
        option_vecs4 = torch.tensor([item.detach().cpu().numpy() for item in option_vecs4]).to(local_device)
        option_vecs5 = torch.tensor([item.detach().cpu().numpy() for item in option_vecs5]).to(local_device)
        option_vecs6 = torch.tensor([item.detach().cpu().numpy() for item in option_vecs6]).to(local_device)
        option_vecs7 = torch.tensor([item.detach().cpu().numpy() for item in option_vecs7]).to(local_device)

        option_similar_vecs = self.option_att(option_vecs0, option_vecs1, option_vecs2, option_vecs3, option_vecs4, option_vecs5, option_vecs6, option_vecs7)

        option_similar = []

        for i in range(len(option_similar_vecs)):
            for j in range(8):
                option_similar.append(option_similar_vecs[i])

        option_similar = torch.tensor([item.detach().cpu().numpy() for item in option_similar]).to(local_device)

        '''
        # csqa
        # compute the commonalities
        option_vecs0 = []
        option_vecs1 = []
        option_vecs2 = []
        option_vecs3 = []
        option_vecs4 = []
        for i in range(int(len(encoder_q) / 5)):
            option_vecs0.append(encoder_qo[i * 5, ..., ...])
            option_vecs1.append(encoder_qo[i * 5 + 1, ..., ...])
            option_vecs2.append(encoder_qo[i * 5 + 2, ..., ...])
            option_vecs3.append(encoder_qo[i * 5 + 3, ..., ...])
            option_vecs4.append(encoder_qo[i * 5 + 4, ..., ...])
        # compute the similarity between choices
        option_vecs0 = torch.tensor([item.detach().cpu().numpy() for item in option_vecs0]).to(local_device)
        option_vecs1 = torch.tensor([item.detach().cpu().numpy() for item in option_vecs1]).to(local_device)
        option_vecs2 = torch.tensor([item.detach().cpu().numpy() for item in option_vecs2]).to(local_device)
        option_vecs3 = torch.tensor([item.detach().cpu().numpy() for item in option_vecs3]).to(local_device)
        option_vecs4 = torch.tensor([item.detach().cpu().numpy() for item in option_vecs4]).to(local_device)

        option_similar_vecs = self.option_att(option_vecs0, option_vecs1, option_vecs2, option_vecs3, option_vecs4)
        option_similar = []

        for i in range(len(option_similar_vecs)):
            for j in range(5):
                option_similar.append(option_similar_vecs[i])
        option_similar = torch.tensor([item.detach().cpu().numpy() for item in option_similar]).to(local_device)

        # contextual information in the question related to commonalities
        s2q, _, ma_s2q, _, sq_weight, _ = self.s2q_att(encoder_q.to(local_device), option_similar, q_mask.to(local_device),
                                               qo_mask.to(local_device))

        # contextual information in the question related to each choice
        o2q, _, ma_o2q, _, oq_weight, _ = self.o2q_att(encoder_q.to(local_device), encoder_qo.to(local_device),
                                               q_mask.to(local_device), qo_mask.to(local_device))

        # the choice-specific representation of question
        hq = o2q - s2q
        semantic_vec = ma_o2q - ma_s2q

        # generate the information
        t5_output = self.t5_model.generate(
            encoder_outputs=ModelOutput(last_hidden_state=semantic_vec.to(local_device)),
            attention_mask=q_mask.to(local_device),
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True
        )
        output_sequences = t5_output.sequences
        output_sequences = output_sequences[:, 1:].contiguous()
        decoder_o = t5_output.decoder_hidden_states
        decoder_o = [item[-1] for item in decoder_o]
        decoder_o = torch.cat(decoder_o, dim=1)

        output_sequences_mask1 = output_sequences != 0
        output_sequences_mask2 = output_sequences != 1
        output_sequences_mask = output_sequences_mask1 * output_sequences_mask2
        output_sequences_mask = output_sequences_mask.long()

        decoder_qo = torch.cat([encoder_qo, decoder_o], dim=1)
        output_sequences_mask = torch.cat([qo_mask, output_sequences_mask], dim=1)

        # enhance the representation of choice
        _, q2a, _, _, _, _ = self.q2a_att(semantic_vec.to(local_device), decoder_qo.to(local_device),
                                          q_mask.to(local_device), output_sequences_mask.to(local_device))

        opt_score = self.option_linear(torch.cat([hq.to(local_device), q2a.to(local_device)], dim=1)).view(-1, self.choice_num)

        return opt_score, output_sequences, sq_weight, oq_weight

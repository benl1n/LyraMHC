import os
import math
import random
import torch
import torch.nn.functional as F
import esm
os.environ["TORCH_HOME"] = "D:/Ben_Plan/paper_factorary/LyraMHC_Project/ESM"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading ESM-2 model on {DEVICE}...")
esm_model, esm_alphabet = esm.pretrained.esm2_t30_150M_UR50D()
esm_model.to(DEVICE)
esm_model.eval()  # 必须 eval 模式
batch_converter = esm_alphabet.get_batch_converter()


BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class ESM_DataProvider:
    def __init__(self, data_file,
                 test_file,
                 train_name,
                 task_name,
                 data_path,
                 use_esm=False,
                 sequence_encode_func=None,
                 batch_size=32,
                 max_len_hla=273,
                 max_len_pep=37,
                 model_count=5,
                 shuffle=True):

        self.use_esm = use_esm
        self.batch_size = batch_size
        self.data_file = data_file
        self.test_file = test_file
        self.sequence_encode_func = sequence_encode_func
        self.shuffle = shuffle
        self.max_len_hla = max_len_hla
        self.max_len_pep = max_len_pep
        self.model_count = model_count

        self.batch_index_train = 0
        self.batch_index_val = 0
        self.batch_index_test = 0

        self.esm_cache = {}

        self.hla_sequence = {}
        # 注意：这里路径你要自己确认对不对
        self.read_hla_sequences(os.path.join(data_path, "proteins_esm.txt"))  # 假设你有这个文件

        self.samples = []
        self.train_samples = []
        self.validation_samples = []
        self.read_training_data()
        self.split_train_and_val()

        self.weekly_samples = []
        self.read_weekly_data()

    def get_esm_repr(self, sequences):
        uncached_seqs = [s for s in sequences if s not in self.esm_cache]

        if uncached_seqs:

            data = [(str(i), seq) for i, seq in enumerate(uncached_seqs)]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(DEVICE)
            # 33
            with torch.no_grad():
                results = esm_model(batch_tokens, repr_layers=[30], return_contacts=False)

            token_reprs = results["representations"][30].cpu()  # 移回 CPU 存内存

            for i, seq in enumerate(uncached_seqs):
                self.esm_cache[seq] = token_reprs[i, 1: 1 + len(seq), :]

        tensors = [self.esm_cache[s] for s in sequences]

        padded = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0)

        # 生成 mask (1为真实数据，0为padding)
        # padded shape: (B, Max_L, 1280)
        mask = torch.zeros((len(tensors), padded.size(1)), dtype=torch.float32)
        for i, t in enumerate(tensors):
            mask[i, :t.size(0)] = 1.0

        return padded, mask


    def read_training_data(self):

        with open(self.data_file) as in_file:
            for line_num, line in enumerate(in_file):
                if line_num == 0:
                    continue

                info = line.strip('\n').split('\t')

                hla_a = info[0]

                if hla_a not in self.hla_sequence:
                    continue

                peptide = info[1]
                if len(peptide) > self.max_len_pep:
                    continue

                ic50 = float(info[2])

                self.samples.append((hla_a, peptide, ic50))

        if self.shuffle:
            random.shuffle(self.samples)

    def split_train_and_val(self):

        vd_count = math.ceil(len(self.samples) / max(self.model_count, 5))
        for i in range(max(self.model_count - 1, 4)):
            self.validation_samples.append(self.samples[i * vd_count:(i + 1) * vd_count])
            temp_sample = self.samples[:]
            del (temp_sample[i * vd_count:(i + 1) * vd_count])
            self.train_samples.append(temp_sample)

        self.validation_samples.append(self.samples[len(self.samples) - vd_count:])
        temp_sample = self.samples[:]
        del (temp_sample[len(self.samples) - vd_count:])
        self.train_samples.append(temp_sample)

    def batch_train(self, order):
        """A batch of training data
        """
        data = self.batch(self.batch_index_train, self.train_samples[order])
        self.batch_index_train += 1
        return data

    def batch_val(self, order):
        """A batch of validation data
        """
        data = self.batch(self.batch_index_val, self.validation_samples[order])
        self.batch_index_val += 1
        return data

    def batch_test(self):
        """A batch of test data
        """
        data = self.batch(self.batch_index_test, self.weekly_samples, testing=True)
        self.batch_index_test += 1
        return data

    def new_epoch(self):
        """New epoch. Reset batch index
        """
        self.batch_index_train = 0
        self.batch_index_val = 0
        self.batch_index_test = 0

    def read_hla_sequences(self, file_path):
        if not os.path.exists(file_path):
            print(f"Warning: HLA file {file_path} not found!")
            return
        with open(file_path, 'r') as in_file:
            for line_num, line in enumerate(in_file):
                if line_num == 0: continue
                info = line.strip('\n').split(' ')
                if len(info) < 2: continue  # 防止空行报错
                seq = info[1]
                self.hla_sequence[info[0]] = seq

    def read_weekly_data(self):

        with open(self.test_file) as in_file:
            for line_num, line in enumerate(in_file):
                if line_num == 0:
                    continue

                info = line.strip('\n').split('\t')
                alleles = info[0]
                peptide = info[1]
                if len(peptide) > self.max_len_pep:
                    continue
                hla_a = alleles
                if hla_a not in self.hla_sequence:
                    continue
                uid = '{hla_a}-{peptide}'.format(
                    hla_a=hla_a,
                    peptide=peptide,
                )
                # print(uid)
                self.weekly_samples.append((hla_a, peptide, uid))


    def train_steps(self):
        return math.ceil(len(self.train_samples[0]) / self.batch_size)

    def val_steps(self):
        return math.ceil(len(self.validation_samples[0]) / self.batch_size)

    def test_steps(self):
        return math.ceil(len(self.weekly_samples) / self.batch_size)


    def batch(self, batch_index, sample_set, testing=False):
        start_i = batch_index * self.batch_size
        current_batch_samples = sample_set[start_i: start_i + self.batch_size]

        # 如果不够 batch_size，补齐
        if len(current_batch_samples) < self.batch_size:
            needed = self.batch_size - len(current_batch_samples)
            # 从已有数据随机抽样补齐，或者从全集抽
            # 简单起见，从当前集循环补
            current_batch_samples += random.choices(sample_set, k=needed)

        # 2. 准备列表
        hla_seqs = []
        pep_seqs = []
        ic50_list = []
        uid_list = []

        for sample in current_batch_samples:
            hla_name = sample[0]
            pep = sample[1]

            # 容错：防止 HLA 名字对不上
            if hla_name not in self.hla_sequence:
                print(f"Error: HLA {hla_name} not found!")
                hla_seq = "A" * self.max_len_hla
            else:
                hla_seq = self.hla_sequence[hla_name]

            if len(hla_seq) > self.max_len_hla: hla_seq = hla_seq[:self.max_len_hla]
            if len(pep) > self.max_len_pep: pep = pep[:self.max_len_pep]

            hla_seqs.append(hla_seq)
            pep_seqs.append(pep)

            if not testing:
                ic50_list.append(sample[2])
            else:
                uid_list.append(sample[2])

        if self.use_esm:
            hla_tensor, hla_mask = self.get_esm_repr(hla_seqs)
            pep_tensor, pep_mask = self.get_esm_repr(pep_seqs)


        else:
            h_t, h_m = [], []
            p_t, p_m = [], []
            for h, p in zip(hla_seqs, pep_seqs):
                # 假设 sequence_encode_func(seq, max_len)
                ht, hm = self.sequence_encode_func(h, self.max_len_hla)
                pt, pm = self.sequence_encode_func(p, self.max_len_pep)
                h_t.append(ht)
                h_m.append(hm)
                p_t.append(pt)
                p_m.append(pm)

            hla_tensor = torch.stack(h_t)
            hla_mask = torch.stack(h_m)
            pep_tensor = torch.stack(p_t)
            pep_mask = torch.stack(p_m)

        hla_tensor = hla_tensor.transpose(1, 2)
        pep_tensor = pep_tensor.transpose(1, 2)

        if not testing:
            return (
                hla_tensor, hla_mask,  # 对应模型的 hla_input
                pep_tensor, pep_mask,  # 对应模型的 pep_input
                torch.tensor(ic50_list),
                current_batch_samples  # validation prototype
            )
        else:
            return (
                hla_tensor, hla_mask,
                pep_tensor, pep_mask,
                uid_list
            )
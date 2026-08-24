# HBUED wrapper, modified from dataset module in TorchEEG.
import os
import glob
import scipy.io
from torcheeg.io import EEGSignalIO, MetaInfoIO

class HBUEDDataset:
    def __init__(self,
                 io_path: str,
                 root_path: str,
                 offline_transform=None,
                 online_transform=None,
                 label_transform=None,
                #  chunk_size=200,
                #  overlap=180,
                 io_size: int = 10485760,
                 io_mode: str = 'lmdb'):

        self.root_path = root_path
        # self.chunk_size = chunk_size
        # self.overlap = overlap

        self.offline_transform = offline_transform
        self.online_transform = online_transform
        self.label_transform = label_transform

        self.io_path = io_path
        os.makedirs(io_path, exist_ok=True)

        # EEG stored in LMDB folder
        self.eeg_io = EEGSignalIO(io_path=io_path, io_size=io_size, io_mode=io_mode)

        # Meta stored in CSV file
        self.info_path = os.path.join(io_path, "info.csv")
        self.info_io = MetaInfoIO(io_path=self.info_path)

        # generate if empty
        if len(self.eeg_io.keys()) == 0:
            print("Generating cache...")
            self._generate_database()

        self.keys = list(self.eeg_io.keys())

    def _generate_database(self):
        # step = self.chunk_size - self.overlap
        files = sorted(glob.glob(os.path.join(self.root_path, "*.mat")))

        for file in files:
            mat = scipy.io.loadmat(file)

            data = mat['data']        # (32, T, 24)
            v_label = mat['v_label']  # (1, 24)
            a_label = mat['a_label']  # (1, 24)

            subject_id = os.path.basename(file).split('_')[0]

            for trial_id in range(data.shape[2]):
                eeg_trial = data[:, :, trial_id]  # (32, T)

                info = {
                    "subject_id": subject_id,
                    "trial_id": trial_id,
                    "valence": float(v_label[0, trial_id]),
                    "arousal": float(a_label[0, trial_id])
                }

                if self.offline_transform is not None:
                    eeg_trial = self.offline_transform(eeg=eeg_trial)['eeg']

                # for start in range(0, eeg_trial.shape[1] - self.chunk_size + 1, step):
                #     chunk = eeg_trial[:, start:start + self.chunk_size]
                #     key = f"{subject_id}_trial{trial_id}_start{start}"

                #     info["key"] = key

                #     self.eeg_io.write_eeg(eeg=chunk, key=key)
                #     self.info_io.write_info(info)

                key = f"{subject_id}_trial{trial_id}"
                info["key"] = key

                self.eeg_io.write_eeg(eeg=eeg_trial, key=key)
                self.info_io.write_info(info)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]

        eeg = self.eeg_io.read_eeg(key)

        # IMPORTANT: meta info is indexed by integer, not key
        info = self.info_io.read_info(index)

        # convert pandas Series to dict
        info = dict(info)

        if self.online_transform is not None:
            eeg = self.online_transform(eeg=eeg)['eeg']

        y = info
        if self.label_transform is not None:
            y = self.label_transform(y=info)['y']

        return eeg, y
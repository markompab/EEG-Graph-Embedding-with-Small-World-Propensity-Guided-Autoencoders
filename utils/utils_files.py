import mne
import pandas as pd


class FileUtils:
    @staticmethod
    def parseNodeFile(srcpath):
        ''' '''
        with open(srcpath, "r") as f:
            lns = f.readlines()
            lns = [ln.replace("\n", "").split(" ") for ln in lns]
            return lns

    @staticmethod
    def parseDuoNodeFile(srcpath):

        with open(srcpath, "r") as f:
            lns = f.readlines()
            rws = []
            for i in range(len(lns)):
                l, r = lns[i].split("|")
                col1 = l.replace("\n", "").split(" ")
                col2 = r.replace("\n", "").split(" ")
                rws.append([col1, col2])

            return rws

    @staticmethod
    def parseTrioNodeFile(srcpath):
         ''' '''
         with open(srcpath, "r") as f:
             lns = f.readlines()
             rws = []
             for i in range(len(lns)):
                 l, m, r = lns[i].split("|")
                 col1 = l.replace("\n", "").split(" ")
                 col2 = m.replace("\n", "").split(" ")
                 col3 = r.replace("\n", "").split(" ")
                 rws.append([col1, col2, col3])

             return rws

    @staticmethod
    def parseQuoNodeFile(srcpath):
         ''' '''
         with open(srcpath, "r") as f:
             lns = f.readlines()
             rws = []
             for i in range(len(lns)):
                 l, m, n, r = lns[i].split("|")
                 col1 = l.replace("\n", "").split(" ")
                 col2 = m.replace("\n", "").split(" ")
                 col3 = n.replace("\n", "").split(" ")
                 col4 = r.replace("\n", "").split(" ")
                 rws.append([col1, col2, col3, col4])

             return rws

    @staticmethod
    def readCsvEeg(srcpath, sfreq=500, maxlen=8500):
        data = pd.read_csv(srcpath)
        #data = data.iloc[:8500]
        if(maxlen > 0):
            data = data.iloc[:maxlen]

        #data = data.iloc[:maxlen]
        ch_names = data.columns.tolist()
        # ch_names =  ['CH 1', 'CH 2', 'CH 3', 'CH 4', 'CH 5', 'CH 6', 'CH 7', 'CH 8', 'CH 9', 'CH 10', 'CH 11', 'CH 12', 'CH 13', 'CH 14', 'CH 15', 'CH 16', 'CH 17', 'CH 18', 'CH 19']
        # sfreq = 10 #500 # Sampling frequency in Hz
        ch_types = ['eeg'] * len(ch_names)
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

        raw = mne.io.RawArray(data.values.transpose(), info)
        raw.set_eeg_reference('average')

        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)

        return raw

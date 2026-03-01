from SoccerNet.Downloader import SoccerNetDownloader as SNdl
mySNdl = SNdl(LocalDirectory="./data")
mySNdl.downloadDataTask(task="mvfouls", split=["train","valid","test","challenge"], password="s0cc3rn3t")
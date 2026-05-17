from SoccerNet.Downloader import SoccerNetDownloader

mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="data/soccernet")

# Les annotations
mySoccerNetDownloader.downloadGames(
    files=["Labels-v2.json"], 
    split=["train", "valid", "test"]
)

# Les features préextraites PCA512
mySoccerNetDownloader.downloadGames(
    files=["1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"], 
    split=["train", "valid", "test"]
)

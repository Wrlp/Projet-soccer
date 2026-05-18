from SoccerNet.Downloader import SoccerNetDownloader
from dotenv import load_dotenv
import os

load_dotenv()

mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="data/soccernet")

# Les annotations
# mySoccerNetDownloader.downloadGames(
#     files=["Labels-v2.json"], 
#     split=["train", "valid", "test"]
# )

# Les features préextraites PCA512
# mySoccerNetDownloader.downloadGames(
#     files=["1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"], 
#     split=["train", "valid", "test"]
# )

# Vidéos 224p 
password = os.getenv("SOCCERNET_PASSWORD")
if password:
    mySoccerNetDownloader.password = password
    mySoccerNetDownloader.downloadGames(
        files=["1_224p.mkv", "2_224p.mkv"],
        split=["train", "valid", "test"],
        verbose=True # afficher bar progression
    )

# from SoccerNet.utils import getListGames

# # Télécharger juste 1 match pour tester
# mySoccerNetDownloader.downloadGame(
#     files=["1_224p.mkv", "2_224p.mkv"],
#     game=getListGames(split="valid")[0]
# )
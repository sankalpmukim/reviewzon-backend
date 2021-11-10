from reviews.reviews import ReviewScraper
from sentiment_analysis.MusicalSentimentAnalysis import SentimentAnalysis_Musical
from sentiment_analysis.LiveSentimentAnalysis import SentimentAnalysis_Live
import pandas as pd
# a = ReviewScraper()
# a.get_pages_data('https://www.amazon.in/Urban-Forest-Detachable-Holder-Wallet/dp/B09HXNMLKP/ref=zg-bs_luggage_1/260-7696386-9517914?pd_rd_w=B7gAd&pf_rd_p=56cde3ad-3235-46d2-8a20-4773248e8b83&pf_rd_r=7HN5497BY01WH1DZGWHY&pd_rd_r=8772838a-3772-466f-b6f2-ae360d085a15&pd_rd_wg=ddB38&pd_rd_i=B09H68SSK5&psc=1', 18)
# livedata = pd.DataFrame(a.all_pages)
# print(livedata.info())
# livedata.to_csv('livedata.csv')

livedata = pd.read_csv('livedata.csv', index_col=0)
SentimentAnalysis_Live(livedata)
# SentimentAnalysis_Musical()

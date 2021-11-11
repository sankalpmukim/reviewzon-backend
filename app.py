from reviews.reviews import ReviewScraper
from sentiment_analysis.MusicalSentimentAnalysis import SentimentAnalysis_Musical
from sentiment_analysis.LiveSentimentAnalysis import SentimentAnalysis_Live
import pandas as pd
# a = ReviewScraper()
# a.get_reviews('https://www.amazon.in/Urban-Forest-Detachable-Holder-Wallet/dp/B09HXNMLKP/ref=zg-bs_luggage_1/260-7696386-9517914?pd_rd_w=B7gAd&pf_rd_p=56cde3ad-3235-46d2-8a20-4773248e8b83&pf_rd_r=7HN5497BY01WH1DZGWHY&pd_rd_r=8772838a-3772-466f-b6f2-ae360d085a15&pd_rd_wg=ddB38&pd_rd_i=B09H68SSK5&psc=1', 40)
# a.get_reviews('https://www.amazon.in/OnePlus-Nord-Blue-128GB-Storage/dp/B097RD2JX8/ref=sr_1_1?crid=3P612QMDFPD7Y&keywords=one%2Bpluse8%2Bnord%2B2&qid=1636620849&s=electronics&sprefix=one%2Bp%2Celectronics%2C296&sr=1-1&th=1', 80)
# a.get_reviews('https://www.amazon.in/New-Apple-iPhone-12-64GB/dp/B08L5VJYV7/ref=sr_1_1_sspa?crid=ZSYDRX5LLSJM&keywords=iphone+12&qid=1636621552&s=electronics&sprefix=iph%2Celectronics%2C301&sr=1-1-spons&psc=1&spLa=ZW5jcnlwdGVkUXVhbGlmaWVyPUEyN0hOVVY5MjhCUjcmZW5jcnlwdGVkSWQ9QTAxOTU5ODIxRjQ1SjQ4TVA3SzE5JmVuY3J5cHRlZEFkSWQ9QTAyOTAwMDU4UFk3SkNXN1dGM0Qmd2lkZ2V0TmFtZT1zcF9hdGYmYWN0aW9uPWNsaWNrUmVkaXJlY3QmZG9Ob3RMb2dDbGljaz10cnVl', 80)
# livedata = pd.DataFrame(a.retrieve_data())
# print(livedata.info())
# livedata.to_csv('livedata.csv')

livedata = pd.read_csv('livedata.csv', index_col=0)
SentimentAnalysis_Live(livedata)
# SentimentAnalysis_Musical()

import asyncio
from aiohttp.client_reqrep import ClientResponse
from lxml.html import fromstring
from selectorlib import Extractor
import json
# from time import sleep
import csv
from dateutil import parser as dateparser
import datetime
import aiohttp
import os


# reviewurl = "https://www.amazon.in/OJOS-Silicone-Compatible-Generation-Protective/product-reviews/B07Y5N98KS/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"


class ReviewScraper:

    def __init__(self) -> None:

        # Create an Extractor by reading from the YAML file
        cwd = os.curdir
        # print(cwd)
        # print(os.getcwd())
        self.e = Extractor.from_yaml_file(
            os.getcwd()+"\\reviews\selectors.yml")

        self.fo = open(os.getcwd()+"\\reviews\config.json")
        self.API_KEYS = json.load(self.fo)["API_KEYS"]
        self.count = 0

    async def scrape(self, r: ClientResponse):
        # Pass the HTML of the page and create
        return (self.e.extract(await r.text()), await r.text())

    def get_proxied_url(self, url: str, count):
        print(count % 4)
        API_KEY = self.API_KEYS[count % 4]
        return f"http://api.scraperapi.com?api_key={API_KEY}&url={url}"

    async def get_page_data(self, session: aiohttp.ClientSession, reviewurl):
        # Download the page using requests
        print("Downloading %s" % reviewurl)
        rows = []
        self.count += 1
        async with session.get(self.get_proxied_url(reviewurl, self.count)) as r:
            data, response_text = await self.scrape(r)
            try:
                if data:
                    # print(data)
                    for r in data['reviews']:
                        r["product"] = data["product_title"]
                        r['url'] = reviewurl
                        if 'verified' in r:
                            # print(r)
                            if r["verified"] and ('Verified Purchase' in r['verified']):
                                r['verified'] = 'Yes'
                            else:
                                r['verified'] = 'No'
                        r['rating'] = r['rating'].split(' out of')[0]
                        date_posted = r['date'].split('on ')[-1]
                        if r['images']:
                            r['images'] = "\n".join(r['images'])
                        r['date'] = dateparser.parse(
                            date_posted).strftime('%d %b %Y')

                        rows.append(r)
            except:
                print("***Data***")
                print(data)
                print("***Response***")
                print(response_text[:25]+'...')
        return rows

    async def main(self, baseurl: str, num_pages: int = 12):
        split = baseurl.split('/')
        print(split)
        reviewurl = split[0]+"//"+split[2]+"/"+split[3]+"/"+"product-reviews"+"/" + \
            split[5]+'/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews'

        async with aiohttp.ClientSession() as session:
            urls = [reviewurl]
            urls += [f"{reviewurl}&pageNumber={i}" for i in range(
                2, num_pages+1)]
            tasks = []
            for url in urls:
                task = asyncio.ensure_future(
                    self.get_page_data(session, url))
                tasks.append(task)

            all_pages = await asyncio.gather(*tasks)
            all_pages = [y for x in all_pages for y in x]
            self.all_pages = all_pages

    def get_pages_data(self, baseurl: str, num_pages: int = 12):
        start_time = datetime.datetime.now()
        asyncio.run(self.main(baseurl, num_pages))
        end_time = datetime.datetime.now()
        print("Total time taken:", end_time-start_time)


if __name__ == "__main__":
    obj = ReviewScraper()
    baseurl = input()
    obj.get_pages_data(baseurl=baseurl)
    print(obj.all_pages)

import asyncio
from typing import List
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

    def __init__(self, logger) -> None:

        # Create an Extractor by reading from the YAML file
        cwd = os.curdir
        # print(cwd)
        # print(os.getcwd())
        try:
            self.e = Extractor.from_yaml_file(
                os.getcwd()+"\\reviews\selectors.yml"
            )
            self.fo = open(os.getcwd()+"\\config.json")
        except FileNotFoundError:
            self.e = Extractor.from_yaml_file(
                os.getcwd()+"/reviews/selectors.yml"
            )
            self.fo = open(os.getcwd()+"/config.json")
        self.API_KEYS = json.load(self.fo)["ScraperAPI"]
        self.count = 0
        self.collected_data = []
        self.flag = True
        self.logger = logger
        self.valid_state = True

    async def scrape(self, r: ClientResponse):
        # Pass the HTML of the page and create
        return (self.e.extract(await r.text()), await r.text())

    def get_proxied_url(self, url: str, count: int):
        total_keys = len(self.API_KEYS)
        API_KEY = self.API_KEYS[count % total_keys]
        return f"http://api.scraperapi.com?api_key={API_KEY}&url={url}"

    async def get_page_data(self, session: aiohttp.ClientSession, reviewurl: str):
        # Download the page using requests
        # print("Downloading %s" % reviewurl)
        rows = []
        self.count += 1
        try:
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
                    if data['next_page'] == None and self.flag:
                        if 'Timed out' not in response_text[:25] and 'Request failed' not in response_text[:25]:
                            self.flag = False
                            self.logger.log(
                                ">No more pages! Stopping further download..", "red")
        except aiohttp.client_exceptions.ClientOSError:
            self.logger.log('>Network error occured', 'red')
        return rows

    async def main(self, baseurl: str, begin_pages: int = 1, num_pages: int = 12):
        split = baseurl.split('/')

        def fixurl(splitted: List[str]) -> List[str]:
            arr = []
            for i in [0, 2, 3, 5]:
                if "?" in splitted[i]:
                    arr.append(splitted[i].split('?')[0])
                else:
                    arr.append(splitted[i])
            return arr
        split = fixurl(split)
        reviewurl = split[0]+"//"+split[1]+"/"+split[2]+"/"+"product-reviews"+"/" + \
            split[3]+'/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews'

        async with aiohttp.ClientSession() as session:
            urls = []
            if begin_pages == 1:
                urls.append(reviewurl)
                urls.extend([f"{reviewurl}&pageNumber={i}" for i in range(
                    2, num_pages+1)])
            else:
                urls.extend([f"{reviewurl}&pageNumber={i}" for i in range(
                    begin_pages, begin_pages+num_pages+1)])
            tasks = []
            for url in urls:
                task = asyncio.ensure_future(
                    self.get_page_data(session, url))
                tasks.append(task)

            all_pages = await asyncio.gather(*tasks)
            all_pages = [y for x in all_pages for y in x]
            self.all_pages = all_pages

    def get_pages_data_unanimous(self, baseurl: str, begin_pages: int = 1, num_pages: int = 12):
        start_time = datetime.datetime.now()
        asyncio.run(
            self.main(baseurl, begin_pages=begin_pages, num_pages=num_pages))
        self.collected_data = self.all_pages
        end_time = datetime.datetime.now()
        print("Total time taken:", end_time-start_time)
        return self.all_pages

    def get_reviews(self, baseurl: str, num_pages: int = 130, begin_pages: int = 1):
        start_time = datetime.datetime.now()
        total_keys = len(self.API_KEYS)
        all_page_data = []
        for i in range(begin_pages, begin_pages+num_pages+1, total_keys*5):
            if self.flag:
                self.logger.log(">Downloading pages "+str(i) +
                                " to "+str(i+(total_keys*5)-1), "lightgreen")
                asyncio.run(
                    self.main(baseurl=baseurl, begin_pages=i,
                              num_pages=total_keys*5-1)
                )
                all_page_data.extend(self.all_pages)
            else:
                # self.logger.log(">No more pages")
                break
        self.collected_data.extend(all_page_data)
        end_time = datetime.datetime.now()
        # self.logger.log("Total time taken:"+ str(end_time-start_time))
        return all_page_data

    def retrieve_data(self):
        return self.collected_data

    def clear_data(self):
        self.collected_data = []


if __name__ == "__main__":
    obj = ReviewScraper()
    baseurl1 = input()
    # baseurl2 = input()
    # obj.get_pages_data_unanimous(baseurl=baseurl, begin_pages=5, num_pages=20)
    obj.get_reviews(baseurl=baseurl1, num_pages=120, begin_pages=1)
    # obj.get_reviews(baseurl=baseurl2, num_pages=80, begin_pages=2)
    print(len(obj.collected_data))

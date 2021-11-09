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

fo = open("config.json")
API_KEYS = json.load(fo)["API_KEYS"]


def get_proxied_url(url: str, count):
    print(count % 4)
    global API_KEYS
    API_KEY = API_KEYS[count % 4]
    return f"http://api.scraperapi.com?api_key={API_KEY}&url={url}"


# Create an Extractor by reading from the YAML file
e = Extractor.from_yaml_file('selectors.yml')


count = 0


async def scrape(r: ClientResponse):

    # Simple check to check if page was blocked (Usually 503)
    # if r.status_code > 500:
    #     if "To discuss automated access to Amazon data please contact" in r.text:
    #         print(
    #             "Page url was blocked by Amazon. Please try using better proxies\n")
    #     else:
    #         print("Page url must have been blocked by Amazon as the status code was %d" % (
    #             r.status_code))
    #     return None
    # print(r.content)
    # Pass the HTML of the page and create
    return (e.extract(await r.text()), await r.text())


# reviewurl = "https://www.amazon.in/OJOS-Silicone-Compatible-Generation-Protective/product-reviews/B07Y5N98KS/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"

async def get_page_data(session: aiohttp.ClientSession, reviewurl):
    # Download the page using requests
    print("Downloading %s" % reviewurl)
    rows = []
    global count
    count += 1
    async with session.get(get_proxied_url(reviewurl, count)) as r:
        data, response_text = await scrape(r)
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
            print(response_text)
    return rows

baseurl = input()
start_time = datetime.datetime.now()


async def main():
    global baseurl
    split = baseurl.split('/')
    print(split)
    reviewurl = split[0]+"//"+split[2]+"/"+split[3]+"/"+"product-reviews"+"/" + \
        split[5]+'/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews'
    with open('data.csv', 'w', encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["title", "content", "date", "variant",
                                                     "images", "verified", "author", "rating", "product", "url"], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        async with aiohttp.ClientSession() as session:
            urls = [reviewurl]
            urls += [f"{reviewurl}&pageNumber={i}" for i in range(2, 13)]
            tasks = []
            for url in urls:
                print(url)
                task = asyncio.ensure_future(get_page_data(session, url))
                tasks.append(task)

            all_pages = await asyncio.gather(*tasks)
        for page in all_pages:
            writer.writerows(page)

asyncio.run(main())
end_time = datetime.datetime.now()
print("Total time taken:", end_time-start_time)

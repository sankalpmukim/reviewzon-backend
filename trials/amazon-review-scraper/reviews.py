from lxml.html import fromstring
from selectorlib import Extractor
import requests
import json
from time import sleep
import csv
from dateutil import parser as dateparser

# Create an Extractor by reading from the YAML file
e = Extractor.from_yaml_file('selectors.yml')


def get_proxies():
    url = 'https://free-proxy-list.net/'
    response = requests.get(url)
    parser = fromstring(response.text)
    proxies = set()
    for i in parser.xpath('//tbody/tr')[:10]:
        if i.xpath('.//td[7][contains(text(),"yes")]'):
            # Grabbing IP and corresponding PORT
            proxy = ":".join([i.xpath('.//td[1]/text()')[0],
                              i.xpath('.//td[2]/text()')[0]])
            proxies.add(proxy)
    return proxies


proxies = get_proxies()
count = 0


def scrape(url):
    headers = {
        'authority': 'www.amazon.com',
        'pragma': 'no-cache',
        'cache-control': 'no-cache',
        'dnt': '1',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (X11; CrOS x86_64 8172.45.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.64 Safari/537.36',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'sec-fetch-site': 'none',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-dest': 'document',
        'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8',
    }

    # Download the page using requests
    print("Downloading %s" % url)
    global count
    print(list(proxies)[count])
    r = requests.get(url, headers=headers, proxies={"http": "http://"+list(
        proxies)[count], "https": "https://"+list(proxies)[count]})
    count += 1
    # Simple check to check if page was blocked (Usually 503)
    if r.status_code > 500:
        if "To discuss automated access to Amazon data please contact" in r.text:
            print(
                "Page %s was blocked by Amazon. Please try using better proxies\n" % url)
        else:
            print("Page %s must have been blocked by Amazon as the status code was %d" % (
                url, r.status_code))
        return None
    print(r.content)
    # Pass the HTML of the page and create
    return e.extract(r.text)


baseurl = input()
split = baseurl.split('/')
print(split)
reviewurl = split[0]+"//"+split[2]+"/"+split[3]+"/"+"product-reviews"+"/" + \
    split[5]+'/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews'
reviewurl = "https://www.amazon.in/OJOS-Silicone-Compatible-Generation-Protective/product-reviews/B07Y5N98KS/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"

with open('data.csv', 'w', encoding="utf-8") as outfile:
    writer = csv.DictWriter(outfile, fieldnames=["title", "content", "date", "variant",
                                                 "images", "verified", "author", "rating", "product", "url"], quoting=csv.QUOTE_ALL)
    writer.writeheader()
    count = 0
    while reviewurl != None:
        data = scrape(reviewurl)
        if data:
            print(data)
            print(count)
            count += 1
            for r in data['reviews']:
                r["product"] = data["product_title"]
                r['url'] = reviewurl
                if 'verified' in r:
                    if 'Verified Purchase' in r['verified']:
                        r['verified'] = 'Yes'
                    else:
                        r['verified'] = 'Yes'
                r['rating'] = r['rating'].split(' out of')[0]
                date_posted = r['date'].split('on ')[-1]
                if r['images']:
                    r['images'] = "\n".join(r['images'])
                r['date'] = dateparser.parse(date_posted).strftime('%d %b %Y')
                writer.writerow(r)
            sleep(5)
            reviewurl = split[0]+"//"+split[2]+data["next_page"]
        else:
            break

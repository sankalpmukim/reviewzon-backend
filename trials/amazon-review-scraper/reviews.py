import requests
from lxml.html import fromstring
from selectorlib import Extractor
import requests
import json
# from time import sleep
import csv
from dateutil import parser as dateparser
import datetime

request_time = datetime.datetime.now()-datetime.datetime.now()


def get_proxied_url(url: str):
    fo = open("config.json")
    API_KEY = json.load(fo)["API_KEY"]

    return f"http://api.scraperapi.com?api_key={API_KEY}&url={url}"


# Create an Extractor by reading from the YAML file
e = Extractor.from_yaml_file('selectors.yml')


count = 0


def scrape(r: requests.Response):

    # Simple check to check if page was blocked (Usually 503)
    if r.status_code > 500:
        if "To discuss automated access to Amazon data please contact" in r.text:
            print(
                "Page url was blocked by Amazon. Please try using better proxies\n")
        else:
            print("Page url must have been blocked by Amazon as the status code was %d" % (
                r.status_code))
        return None
    # print(r.content)
    # Pass the HTML of the page and create
    return (e.extract(r.text), r.text)


baseurl = input()
start_time = datetime.datetime.now()
split = baseurl.split('/')
print(split)
reviewurl = split[0]+"//"+split[2]+"/"+split[3]+"/"+"product-reviews"+"/" + \
    split[5]+'/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews'
# reviewurl = "https://www.amazon.in/OJOS-Silicone-Compatible-Generation-Protective/product-reviews/B07Y5N98KS/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"

with open('data.csv', 'w', encoding="utf-8") as outfile:
    writer = csv.DictWriter(outfile, fieldnames=["title", "content", "date", "variant",
                                                 "images", "verified", "author", "rating", "product", "url"], quoting=csv.QUOTE_ALL)
    writer.writeheader()
    count = 1
    while reviewurl != None and count <= 12:

        # Download the page using requests
        print("Downloading %s" % reviewurl)
        start_request = datetime.datetime.now()
        r = requests.get(get_proxied_url(reviewurl))
        end_request = datetime.datetime.now()
        request_time += (end_request-start_request)
        data, response_text = scrape(r)
        try:
            if data:
                # print(data)
                print(count)
                count += 1
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
                    writer.writerow(r)
                # sleep(5)
                if data["next_page"] == None:
                    print("All pages exhausted")
                    break
                reviewurl = split[0]+"//"+split[2]+data["next_page"]
            else:
                print("Yeeted")
                print(data)
                break
        except:
            print("***Data***")
            print(data)
            print("***Response***")
            print(response_text)


end_time = datetime.datetime.now()
print("Total time taken:", end_time-start_time)
print("Time in API requests:", request_time)

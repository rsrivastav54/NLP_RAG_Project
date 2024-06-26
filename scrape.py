import pandas as pd
from urllib.request import Request, urlopen
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import ssl
import trafilatura
import json
from tqdm import tqdm

SITEMAP_URL = "http://localhost:8000/sitemap2.xml"
OUTPUT_FILE = "scraped_data.txt"
def get_sitemap(url):
    req = Request(
        url=url,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    #print("request encoding :", req.encoding)
    response = urlopen(req)
    xml = BeautifulSoup(
        response, 
        "lxml-xml", 
        from_encoding=response.info().get_param("charset")
    )
    print(xml.encoding)
    return xml


def sitemap_to_dataframe(xml, name=None, data=None, verbose=False):
    df = pd.DataFrame(columns=["loc", "changefreq", "priority", "domain", "sitemap_name"])
    urls = xml.find_all("url")
    for url in urls:
        if xml.find("loc"):
            loc = url.findNext("loc").text
            parsed_uri = urlparse(loc)
            domain = "{uri.netloc}".format(uri=parsed_uri)
        else:
            loc = ""
            domain = ""
        if xml.find("changefreq"):
            changefreq = url.findNext("changefreq").text
        else:
            changefreq = ""
        if xml.find("priority"):
            priority = url.findNext("priority").text
        else:
            priority = ""
        if name:
            sitemap_name = name
        else:
            sitemap_name = ""
        row = {
            "domain": domain,
            "loc": loc,
            "changefreq": changefreq,
            "priority": priority,
            "sitemap_name": sitemap_name,
        }
        if verbose:
            print(row)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df


def extract_text_from_url(url):
    downloaded_url = trafilatura.fetch_url(url)
    extracted = trafilatura.extract(
        downloaded_url, 
        output_format="json", 
        with_metadata=True, 
        include_comments = False,
        date_extraction_params={"extensive_search": True, "original_date": True}
    ).encode("utf-8")
    json_output = json.loads(extracted)
    return json_output["text"]
        
def main():
    ssl._create_default_https_context = ssl._create_stdlib_context
    xml = get_sitemap(SITEMAP_URL)
    print("Encoding:", xml.encoding)

    df = sitemap_to_dataframe(xml, verbose=False)
    urls = df["loc"].to_numpy()
    urls = [url for url in urls if "%" not in url]

    with open(OUTPUT_FILE, "w", encoding='utf-8') as f:
        for url in tqdm(urls[:]):
            topic = url.split("/")[-1]
            # print(topic)
            try:
                text = extract_text_from_url(url=url)
                text = text.lower()
                text = text.replace("key takeaways from this chapter", "")
                text = text.replace("we recommend reading this chapter on varsity to learn more and understand the concepts in-depth.", "")
                text = text.replace("varsity", "")
                f.writelines(topic + ": \n")
                f.writelines(text  + "\n###\n")
            except Exception as e:
                print(topic)
                print(e)

if __name__ == "__main__":
    main()
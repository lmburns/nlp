# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Cat
#     language: python
#     name: cat
# ---

# +
# =======================================================
# NOTE: Piping in Python (similar to %>% in R, | in Unix)
from datetime import datetime
from itertools import chain
import math
import re
import ssl

from bs4 import BeautifulSoup as bs
from fn import F, _
from matplotlib import test
import numpy as np
import pandas as pd
import requests
from sspipe import p, px
from toolz import pipe
import tabloo

# import dfply                            # like dplyr

pipe(12, math.sqrt, str)  # toolz
12 | p(math.sqrt) | px ** 2 | p(str)  # sspipe
(F(math.sqrt) >> _ ** 2 >> str)(12)  # fn
# -


# +
url = "https://slashdot.org/"

headers = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    "accept-encoding": "gzip, deflate, br",
    "accept-language": "en-US,en;q=0.9,es;q=0.8",
    # 'cache-control': 'max-age=0',
    "dnt": "1",
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "cross-site",
    "sec-fetch-user": "?1",
    "sec-gpc": "1",
    "upgrade-insecure-requests": "1",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.101 Safari/537.36",
}

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

r = requests.get(url, headers=headers)
soup = bs(r.text, "html.parser")

#### ┌─────────────────────────────────────────────────┐
#### │ Scraping one page                               │
#### └─────────────────────────────────────────────────┘

# +
# Title, link, rating
tit = [t.get_text() for t in soup.select("h2 span a")]

# Title of post
title = [j for i in soup.find_all("span", class_="story-title") for j in i.find("a")]

# Date/time of post
dated = [d.get_text() for d in soup.select("span.story-byline time")]
date = [re.sub("on|@", "", x).strip() for x in dated]
dt = [datetime.strptime(d, "%A %B %d, %Y %I:%M%p") for d in date]

# External link to post
elink = [l.text.strip() for l in soup.select("h2 span span")]

# Comments on post
comments = (
    [x.get_text() for x in soup.select("span.comment-bubble a")]
    | p(np.array)
    | px.astype("int")
)

# Category of post
cat = [b.get("alt") for b in soup.find_all("img")] | p(list, p(filter, None, px))
# Using this sort of as a try except in case it were to not exist
category = [x.replace("Icon", "") for x in cat] | p(filter, None) | p(list)

# User who made the post
user = [
    u.get_text(" ", strip=True).replace("\n", "").replace("\t", "")
    for u in soup.select("span.story-byline")
]
user = [" ".join(a.split()) | p(re.findall, r"Postedby\s(\w+)", px) for a in user]

# Popularity of post (ratings? red?)
pop = [
    re.findall("'([a-zA-Z0-9,\s]*)'", prop["onclick"]) | px[1]
    for prop in soup.find_all("span", attrs={"alt": "Popularity"})
]
# -

#### ┌─────────────────────────────────────────────────┐
#### │ Dataframe                                       │
#### └─────────────────────────────────────────────────┘

# +
df = pd.DataFrame(
    {
        "title": title,
        "date": dt,
        "exlink": elink,
        "comments": comments,
        "category": category,
        "user": user,
        "popular": pop,
    }
)


tabloo.show(df)
# from operator import itemgetter
# df['user'] = df['user'] | p(map, p(itemgetter(0)), px) | p(list)

df["user"] = df["user"] | p(chain.from_iterable, px) | p(list)
df["exlink"] = df["exlink"] | px.str.replace(r"\(|\)", "", regex=True)
# -

#### ┌─────────────────────────────────────────────────┐
#### │ Functions for scraping more than one page       │
#### └─────────────────────────────────────────────────┘

# +
import time

def get_page(url):
    response = requests.get(url)
    if not response.ok:
        print("Server Responded: ", response.status_code)
    else:
        soup = bs(response.text, "lxml")
        time.sleep(3)
    return soup


def get_data(soup):
    try:
        title = [j for i in soup.find_all("span", class_="story-title") for j in i.find("a")]
    except:
        title = ""

    try:
        dated = [d.get_text(" ", strip=True) for d in soup.select("span.story-byline time")]
        date = [re.sub("on|@", "", x).strip() for x in dated]
        dt = [datetime.strptime(d, "%A %B %d, %Y %I:%M%p") for d in date]
    except:
        dt = ""

    try:
        pattern = re.compile(r"\([a-z0-9.\-]+[.](\w+)\)")
        curls = {}

        for idx, u in enumerate(ex):
            if not pattern.search(u):
                curls[idx] = "Empty"
            else:
                curls[idx] = pattern.search(u).group()

        elink = list(curls.values())
    except:
        elink = ""

    try:
        comments = (
            [x.get_text() for x in soup.select("span.comment-bubble a")]
            | p(np.array)
            | px.astype("int"))
    except:
        comments = ""

    try:
        cat = ([b.get("alt") for b in soup.find_all("img")]
              | p(list, p(filter, None, px)))
        category = ([x.replace("Icon", "") for x in cat]
                    | p(filter, None)
                    | p(list))
    except:
        category = ""

    try:
        user = [u.get_text(" ", strip=True).replace("\n", "").replace("\t", "")
                for u in soup.select("span.story-byline")]
        user = [" ".join(a.split())
                | p(re.findall, r"Postedby\s(\w+)", px) for a in user]
    except:
        user = ""

    try:
        pop = [re.findall("'([a-zA-Z0-9,\s]*)'", prop["onclick"]) | px[1]
            for prop in soup.find_all("span", attrs={"alt": "Popularity"})]
    except:
        pop = ""

    temp = pd.DataFrame(
        {
            "title": title,
            "date": dt,
            "exlink": elink,
            "comments": comments,
            "category": category,
            "user": user,
            "popular": pop
        }
    )

    return temp
# -

#### ┌─────────────────────────────────────────────────┐
#### │ Scraping more than one page                     │
#### └─────────────────────────────────────────────────┘

# +
from random import sample

base_url = "https://slashdot.org/?page="
urls = [base_url + str(i) for i in range(1, 100)]
test_u = urls | p(sample, 2)


data = [get_data(get_page(x)) for x in test_u] | p(pd.concat, px)



data
# -

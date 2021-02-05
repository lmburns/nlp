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
from toolz import pipe
# import dfply                          # similar to dplyr / has a pipe
from sspipe import p, px
from fn import F, _
import math

pipe(12, math.sqrt, str)                # toolz
12 | p(math.sqrt) | px ** 2 | p(str)    # sspipe
(F(math.sqrt) >> _**2 >> str)(12)       # fn
# =======================================================

# +
from bs4 import BeautifulSoup as bs
import requests
import ssl
import re
from datetime import datetime

import numpy as np
import pandas as pd

# +
url = "https://slashdot.org/"

headers = {
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'accept-encoding': 'gzip, deflate, br',
    'accept-language': 'en-US,en;q=0.9,es;q=0.8',
    #'cache-control': 'max-age=0',
    'dnt': '1',
    'sec-fetch-dest': 'document',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-site': 'cross-site',
    'sec-fetch-user': '?1',
    'sec-gpc': '1',
    'upgrade-insecure-requests': '1',
    'user-agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.101 Safari/537.36'
    }

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

r = requests.get(url, headers=headers)
soup = bs(r.text, 'html.parser')

# +
### Title, link, rating
tit = [t.get_text() for t in soup.select('h2 span a')]

### Title of post
title = [j for i in soup.find_all('span', class_='story-title') for j in i.find('a')]

### Date/time of post
dated = [d.get_text() for d in soup.select('span.story-byline time')]
date = [re.sub('on|@', '', x).strip() for x in dated]
dt = [datetime.strptime(d, "%A %B %d, %Y %I:%M%p") for d in date]

### External link to post
elink = [l.text.strip() for l in soup.select('h2 span span')]

### Votes on post
votes = [v.text.strip() for v in soup.find_all('span', class_='comment-bubble')]

### Category of post
classification = [b.get('alt') for b in soup.find_all('img')]
c = list(filter(None, classification))
# Using this sort of as a try except in case it were to not exist
category = [x.replace('Icon', '') for x in c] | p(filter, None) | p(list)

### User who made the post
user = [u.get_text() for u in soup.select('span.story-byline a')]

aa = [u.get_text() for u in soup.select('span.story-byline')]
"Posted" in aa

import grep

len(aa)

### Popularity of post (ratings? red?)
onclick = [re.findall("'([a-zA-Z0-9,\s]*)'", prop['onclick']) for prop in soup.find_all("span", attrs={"alt":"Popularity"})]
pop = [p[1] for p in onclick]
# -



# +
pd.DataFrame({'title': title, 'date': dt, 'exlink': elink, 'category': category, 'user': user, 'popular': pop})

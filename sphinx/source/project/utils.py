
import pandas as pd
import pathlib
import requests
import bs4
import re
import os
import numpy as np
import datetime
from PIL import Image
import random
import time

from bokeh.plotting import figure
from bokeh.models import HoverTool

datadir = pathlib.Path(__file__).parent.parent.parent.parent / 'data'


def get_rtd_analytics_data(project):
    filenames = sorted(datadir.glob(project + "*"))
    dfs = [pd.read_csv(fn, parse_dates=True) for fn in filenames]
    df = pd.concat(dfs)
    df['Date'] = pd.to_datetime(df['Date'])
    # the last day in a file can be incomplete, so in the overlap range we
    # prefer newer records (indicated by occurring later in `filenames`)
    df = df.drop_duplicates(['Date', 'Version', 'Path'], keep='last')
    df = df.drop(columns=['Unnamed: 0'])
    df = df.sort_values(['Version', 'Date', 'Path'])
    # prefix X.Y.Z with "v"
    is_bare_version = df['Version'].str.contains('.', regex=False) & ~df['Version'].str.startswith('v')
    df.loc[is_bare_version, 'Version'] = 'v' + df.loc[is_bare_version, 'Version']
    # remove PR builds and such
    keep = (
        df['Version'].str.startswith('v') |
        df['Version'].isin(['stable', 'latest']) |
        (df['Version'] == '0.1')
    )
    df = df.loc[keep, :]
    df = df.reset_index(drop=True)
    return df


user = os.getenv('GH_USERNAME')
token = os.getenv('GH_TOKEN')
auth = (user, token)


def _fetch_gh_api(repo, page=None):
    '''return API json response, plus the max page count'''
    url = f'https://api.github.com/repos/{repo}/stargazers'
    if page is not None:
        url += f"?page={page}"
    #print(url)
    # necessary to get starred_at data:
    headers = {'Accept': 'application/vnd.github.v3.star+json'}
    response = requests.get(url, headers=headers, auth=auth)

    data = response.json()
    try:
        link_text = response.headers['link']
        matches = re.findall(r'page=(\d*)', link_text)
        page_numbers = map(lambda s: int(s.split("=")[-1]), matches)
        N = max(page_numbers)
    except KeyError:
        N = 1
    return data, N


def get_github_stars(repo):
    data, N = _fetch_gh_api(repo)
    for i in range(2, N+1):
        data.extend(_fetch_gh_api(repo, i)[0])

    star_date = [d['starred_at'] for d in data]
    user_name = [d['user']['login'] for d in data]
    df = pd.DataFrame({'star_date': star_date, 'user_name': user_name})
    df['star_date'] = pd.to_datetime(df['star_date'])
    df = df.sort_values('star_date')
    return df


def plot_github_stars_timeseries(gh):
    star_curve = gh.set_index('star_date').assign(x=1).x.cumsum().resample('d').max()
    # project out to present:
    star_curve = pd.concat([star_curve, pd.Series({datetime.datetime.utcnow(): np.nan})])
    star_curve = star_curve.ffill()

    p = figure(height=350, x_axis_type="datetime")
    hover_tool = HoverTool(tooltips=[('Date', '@x{%Y-%m-%d}'), ('Total Stars', '@y')],
                           formatters={'@x': 'datetime'})
    hover_tool.point_policy = 'snap_to_data'
    p.add_tools(hover_tool)

    p.line(star_curve.index, star_curve)
    p.yaxis.axis_label = 'Total Stars'
    p.xaxis.axis_label = 'Date'
    return p


def get_github_contributors(repo):
    url = f"https://api.github.com/repos/{repo}/contributors?per_page=100"
    headers = {'Accept': 'application/vnd.github+json'}
    auth = (user, token)
    response = requests.get(url, headers=headers, auth=auth)
    contributor_data = response.json()

    if len(contributor_data) == 100:
        raise ValueError("You need to generalize this code to handle multiple pages")

    return contributor_data


def make_github_contributors_mosaic(contributor_data, n_wide=None, n_high=None):
    images = []
    for contributor in contributor_data:
        response = requests.get(contributor['avatar_url'], stream=True)
        images.append(Image.open(response.raw))

    # randomize order
    random.shuffle(images)

    widths, heights = zip(*(i.size for i in images))

    if n_wide is None:
        n_wide = 10
        n_high = len(images) // n_wide + 1

    image_width = 60
    image_height = 60
    buffer = 5

    total_width = image_width * n_wide + buffer * (n_wide-1)
    total_height = image_height * n_high + buffer * (n_high-1)

    new_im = Image.new('RGBA', (total_width, total_height), color=(255, 255, 255))

    x_offset = 0
    y_offset = 0
    for i, im in enumerate(images):
        im = im.resize((image_width, image_height))
        new_im.paste(im, (x_offset, y_offset))
        x_offset += image_width + buffer
        if (i % n_wide) == (n_wide - 1):
            x_offset = 0
            y_offset += image_height + buffer

    return new_im


def get_github_contributor_timeseries(repo, max_retries=10, retry_delay=10):
    url = f"https://github.com/{repo}/graphs/contributors-data"
    
    # I don't 100% understand this, but I think if this data hasn't been
    # regenerated in a while, the initial request comes back empty while
    # the data gets refreshed on the server side.  You just have to poll it
    # until you get something back.
    for _ in range(max_retries):
        response = requests.get(url, headers={"Accept": "application/json"})
        try:
            js = response.json()
        except:
            time.sleep(retry_delay)
            pass
        else:
            break
    else:
        raise Exception("Could not fetch contributor data")

    first_commit_dates = []
    for record in js:
        username = record['author']['login']
        weekdata = record['weeks']
        first = min([x for x in weekdata if x['c'] > 0], key=lambda x: x['w'])
        ts = pd.to_datetime(first['w'], unit='s')
        first_commit_dates.append(ts)
    
    s = pd.Series(1, index=first_commit_dates)
    out = s.sort_index().resample('d').sum().replace(0, np.nan).cumsum().ffill()
    return out


def plot_github_contributors_timeseries(gh):
    # project out to present:
    gh = pd.concat([gh, pd.Series({datetime.datetime.utcnow(): np.nan})])
    gh = gh.ffill()

    p = figure(height=350, x_axis_type="datetime")
    hover_tool = HoverTool(tooltips=[('Date', '@x{%Y-%m-%d}'), ('Total Contributors', '@y')],
                           formatters={'@x': 'datetime'})
    hover_tool.point_policy = 'snap_to_data'
    p.add_tools(hover_tool)

    p.line(gh.index, gh)
    p.yaxis.axis_label = 'Total Contributors'
    p.xaxis.axis_label = 'Date'
    return p


def fetch_gs_citations(publication_id):
    # TODO: this function is rather buggy, for example it sometimes returns
    # other junk instead of the year.  it would be better (but probably
    # much slower) to parse the bibtex entry for each citation.

    url = f'https://scholar.google.com/scholar?hl=en&as_sdt=4005&sciodt=0,6&cites={publication_id}&scipsc='
    
    results = []
    
    while True:

        response = requests.get(url)
        if b'block will expire' or b"verify that you're not a robot" in response.content:
            raise Exception('rate limit')

        soup = bs4.BeautifulSoup(response.content, features='lxml')

        page_records = soup.find_all(attrs={'class': 'gs_r gs_or gs_scl'})
        for record in page_records:
            result = {}
    
            title_element = record.find(attrs={'class': 'gs_rt'})
            title_link = title_element.find('a')
            if title_link is not None:
                result['title'] = title_link.text
                result['url'] = title_link.attrs['href']
            else:
                result['title'] = title_element.find_all('span')[-1].text
                result['url'] = None
    
            author_element = record.find(attrs={'class': 'gs_a'})
            fields = re.split("\s- ", author_element.text)
            if len(fields) == 3:
                authors, journal_year, _ = fields
                if "," in journal_year:
                    journal, year = journal_year.rsplit(", ", 1)
                else:
                    year = journal_year
                    journal = None
            else:
                authors = fields[0]
                journal = None
                year = None
    
            result['authors'] = authors
            result['year'] = year
            result['journal'] = journal
    
            results.append(result)
    
        next_page_nav = soup.find(attrs={'class': 'gs_ico gs_ico_nav_next'})
        if next_page_nav is None:
            break
    
        url = 'https://scholar.google.com' + next_page_nav.parent.attrs['href']
        time.sleep(2)

    df = pd.DataFrame(results)
    return df

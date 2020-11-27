from pyyoutube import Api
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities  
import pandas as pd
import requests
import json
import time


# Code is partially grabbed from this repository:
# https://github.com/egbertbouman/youtube-comment-downloader

class YScraper():
    
    def search_dict(self, partial, key):
        """
        A handy function that searches for a specific `key` in a `data` dictionary/list
        """
        if isinstance(partial, dict):
            for k, v in partial.items():
                if k == key:
                    # found the key, return the value
                    yield v
                else:
                    # value of the dict may be another dict, so we search there again
                    for o in self.search_dict(v, key):
                        yield o
        elif isinstance(partial, list):
            # if the passed data is a list
            # iterate over it & search for the key at the items in the list
            for i in partial:
                for o in self.search_dict(i, key):
                    yield o
    
    
    def find_value(self, html, key, num_sep_chars=2, separator='"'):
        # define the start position by the position of the key + 
        # length of key + separator length (usually : and ")
        start_pos = html.find(key) + len(key) + num_sep_chars
        # the end position is the position of the separator (such as ")
        # starting from the start_pos
        end_pos = html.find(separator, start_pos)
        # return the content in this range
        return html[start_pos:end_pos]
    
    
    def get_comments(self, url):
        session = requests.Session()
        # make the request
        res = session.get(url)
        # extract the XSRF token
        xsrf_token = self.find_value(res.text, "XSRF_TOKEN", num_sep_chars=3)
        # parse the YouTube initial data in the <script> tag
        data_str = self.find_value(res.text, 'window["ytInitialData"] = ', num_sep_chars=0, separator="\n").rstrip(";")
        # convert to Python dictionary instead of plain text string
        data = json.loads(data_str)
        # search for the ctoken & continuation parameter fields
        pagination_data = None
        continuation_tokens = None
        for r in self.search_dict(data, "itemSectionRenderer"):
            try:
                pagination_data = next(self.search_dict(r, "nextContinuationData"))
            except StopIteration:
                pass
            if pagination_data:
                # if we got something, break out of the loop,
                # we have the data we need
                break
        if pagination_data is not None:
            continuation_tokens = [(pagination_data['continuation'], pagination_data['clickTrackingParams'])]
        else:
            print("UNABLE TO RETRIEVE COMMENTS")
    
        while continuation_tokens:
            # keep looping until continuation tokens list is empty (no more comments)
            continuation, itct = continuation_tokens.pop()
        
            # construct params parameter (the ones in the URL)
            params = {
                "action_get_comments": 1,
                "pbj": 1,
                "ctoken": continuation,
                "continuation": continuation,
                "itct": itct,
            }
    
            # construct POST body data, which consists of the XSRF token
            data = {
                "session_token": xsrf_token,
            }
    
            # construct request headers
            headers = {
                "x-youtube-client-name": "1",
                "x-youtube-client-version": "2.20200731.02.01"
            }
    
            # make the POST request to get the comments data
            response = session.post("https://www.youtube.com/comment_service_ajax", params=params, data=data, headers=headers)
            # convert to a Python dictionary
            try:
                
                comments_data = json.loads(response.text)
        
                for comment in self.search_dict(comments_data, "commentRenderer"):
                    # iterate over loaded comments and yield useful info
                    yield {
                        "commentId": comment["commentId"],
                        "text": ''.join([c['text'] for c in comment['contentText']['runs']]),
                        "time": comment['publishedTimeText']['runs'][0]['text'],
                        "isLiked": comment["isLiked"],
                        "likeCount": comment["likeCount"],
                        # "replyCount": comment["replyCount"],
                        'author': comment.get('authorText', {}).get('simpleText', ''),
                        'channel': comment['authorEndpoint']['browseEndpoint']['browseId'],
                        'votes': comment.get('voteCount', {}).get('simpleText', '0'),
                        'photo': comment['authorThumbnail']['thumbnails'][-1]['url'],
                        "authorIsChannelOwner": comment["authorIsChannelOwner"],
                    }
        
                # load continuation tokens for next comments (ctoken & itct)
                continuation_tokens = [(next_cdata['continuation'], next_cdata['clickTrackingParams'])
                                 for next_cdata in self.search_dict(comments_data, 'nextContinuationData')] + continuation_tokens
            except json.JSONDecodeError:
                pass
            # avoid heavy loads with popular videos
            time.sleep(0.1)
            
    
    def scrape_youtube_search(self, api_key, query, firefox_path, driver_path, filter_):  
        '''
        Gets the links of the videos
        - Function to scrape every link of a youtube search(for a given page that has been fully loaded). 
        - In query_and_labels dataframe resides every query.
        - To achieve the scraping, we use selenium (a firefox add-on that has been originally created for website testing) and BeautifulSoup
        '''
        cap                 = DesiredCapabilities.FIREFOX.copy() 
        options             = webdriver.FirefoxOptions()
        options.binary      = firefox_path
        cap["marionette"]   = True
        driver              = webdriver.Firefox(options=options,executable_path=r''+driver_path, capabilities=cap)
        api                 = Api(api_key=api_key)
        urls                = []
        channels_name       = []
        channels_id         = []
        channels_country    = []
        videos_title        = []
        videos_views        = []
        upload_dates        = []
        for value in query.itertuples():     
            try:
                print("QUERY {} IS BEING TREATED...".format(value.query))
                driver.get(f"https://www.youtube.com/results?q={value.query}&sp={filter_}&gl={value.country}")
                html = driver.page_source
                soup = BeautifulSoup(html, features="html.parser")
                main_tags = soup.find_all("div", id="dismissable")
                for main_tag in main_tags:              
                        url = main_tag.find("a", id="thumbnail")["href"]
                        urls.append(url)
                        channel_name = main_tag.find("paper-tooltip", class_="ytd-channel-name").text.strip()
                        channels_name.append(channel_name)
                        channel_id = main_tag.find("div", id="channel-info").find("a")["href"].split("/")[2]
                        channels_id.append(channel_id)
                        video_title = main_tag.find("a", id="video-title")["title"]
                        videos_title.append(video_title)
                        video_views = main_tag.find("div", id="metadata-line").find_all("span")[0].text
                        videos_views.append(video_views)      
                        upload_date = main_tag.find("div", id="metadata-line").find_all("span")[1].text
                        upload_dates.append(upload_date)
                        channel_info = api.get_channel_info(channel_id=channel_id)
                        country="unknown"
                        if channel_info.items:
                            country = channel_info.items[0].to_dict()["brandingSettings"]["channel"]["country"]        
                        channels_country.append(country)    
            except (WebDriverException, UnicodeEncodeError):
                print(f"Query not resolved: {value.query}")
                pass   
        print("DONE!")
        frame = pd.DataFrame({"channel_id":channels_id, 
                              "channel_name":channels_name, 
                              "channel_country":channels_country, 
                              "url":urls,
                              "video_title":videos_title,
                              "video_views":videos_views,
                              "upload_date":upload_dates})
        return frame.drop_duplicates().reset_index(drop=True)
    
    
    def scrape_youtube_comments(self, urls, n_comments, path_to_save=None):
        """
        Gets the comments for every single video link
        """
        all_comments = []
        for item in urls.itertuples():
            url = item.url
            index = item.Index
            link = f"https://www.youtube.com{url}"
            print(f"{index} - RETRIEVING {n_comments} COMMENT(S) FROM URL: {link}...")
            try:
                for count, comment in enumerate(self.get_comments(link)):
                    if count == n_comments:  
                        break
                    all_comments.append(comment)
            except json.JSONDecodeError:
                pass
        print("DONE!")
        comments_frame = pd.DataFrame(all_comments)
        #saving dataframe
        comments_frame.to_csv(path_to_save)
        return comments_frame
   


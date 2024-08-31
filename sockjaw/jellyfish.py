# modified Jellyfish code to make HTTP requests through the expecc
# route, directly accessing the internet. it uses the requests library
# in Python to make HTTP requests.

# run this modified Jellyfish code in one terminal and the
# HTTP proxy server code in another terminal to see the
# Jellyfish navigating through the internet.

import requests
from bs4 import BeautifulSoup
import random

visited_urls = set()

def crawl(url, depth=1, max_depth=3):
    if depth > max_depth:
        return

    try:
        if url in visited_urls:
            print(f"Already visited: {url}")
            return

        visited_urls.add(url)

        response = requests.get(url)
        if response.status_code == 200:
            print(f"Crawling: {url}")

            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.find_all('a', href=True)

            for link in links:
                next_url = link['href']
                if next_url.startswith('http'):
                    crawl(next_url, depth + 1, max_depth)

    except Exception as e:
        print(f"Error crawling {url}: {e}")

if __name__ == "__main__":

    desired_choices = [
        "http://ajaxian.com",
        "http://digg.com",
        "http://en.wikipedia.org",
        "http://english.aljazeera.net/HomePage",
        "https://online.wellsfargo.com",
        "http://de.wikipedia.org",
        "http://login.yahoo.com",
        "http://mail.google.com",
        "mail.yahoo.com",
        "http://my.yahoo.com",
        "http://reddit.com",
        "http://seoblackhat.com",
        "http://slashdot.org",
        "http://techfoolery.com",
        "http://weblogs.asp.net/jezell",
        "http://www.amazon.com",
        "http://www.aol.com",
        "http://www.apple.com",
        "http://www.ask.com",
        "http://www.bankofamerica.com",
        "http://www.bankone.com",
        "http://www.bing.com",
        "http://www.blackhat.com",
        "http://www.blogger.com",
        "http://www.bloglines.com",
        "http://www.bofa.com",
        "http://www.capitalone.com",
        "http://www.cenzic.com",
        "http://www.cgisecurity.com",
        "http://www.chase.com",
        "http://www.citibank.com",
        "http://www.comerica.com",
        "http://www.craiglist.org",
        "http://www.e-gold.com",
        "http://www.ebay.com",
        "http://www.etrade.com",
        "http://www.expedia.com",
        "http://www.flickr.com",
        "http://www.go.com",
        "http://www.google.ca",
        "http://www.google.com",
        "http://www.google.com.br",
        "http://www.google.ch",
        "http://www.google.cn",
        "http://www.google.co.jp",
        "http://www.google.de",
        "http://www.google.es",
        "http://www.google.fr",
        "http://www.hattrick.org",
        "http://www.hsbc.com",
        "http://www.hi5.com",
        "http://www.icq.com",
        "http://www.imdb.com",
        "http://www.imageshack.us",
        "http://www.jailbabes.com",
        "http://www.linkedin.com",
        "http://www.tagged.com",
        "http://www.globo.com",
        "http://www.megavideo.com",
        "http://www.kino.to",
        "http://www.google.ru",
        "http://www.friendster.com",
        "http://www.nytimes.com",
        "http://www.megaclick.com",
        "http://www.mininova.org",
        "http://www.thepiratebay.org",
        "http://www.netlog.com",
        "http://www.orkut.com",
        "http://www.megacafe.com",
        "http://www.answers.com",
        "http://www.scribd.com",
        "http://www.xing.com",
        "http://www.partypoker.com",
        "http://www.skype.com",
        "http://mail.google.com",
        "http://www.microsoft.com",
        "http://www.msn.com",
        "http://www.myspace.com",
        "http://www.ntobjectives.com",
        "http://www.passport.net",
        "http://www.paypal.com",
        "http://www.photobucket.com",
        "http://www.pornhub.com",
        "http://www.reddit.com",
        "http://www.redtube.com",
        "http://www.sourceforge.net",
        "http://www.spidynamics.com",
        "http://www.statefarm.com",
        "http://www.twitter.com",
        "http://www.vkontakte.ru",
        "http://www.wachovia.com",
        "http://www.wamu.com",
        "http://www.watchfire.com",
        "http://www.webappsec.org",
        "http://www.wellsfargo.com",
        "http://www.whitehatsec.com",
        "http://www.wordpress.com",
        "http://www.xanga.com",
        "http://www.yahoo.com",
        "http://seoblackhat.com",
        "http://www.alexa.com",
        "http://www.youtube.com",
        "https://banking.wellsfargo.com",
        "https://commerce.blackhat.com",
    ]

    random_choice = random.choice(desired_choices)
    start_url = random_choice

    crawl(start_url)

# This code will now crawl web pages starting from the specified
# start_url crawls through the links on the provided webpage up
# to a certain depth (max_depth). adjusting the max_depth parameter
# controls how deep the crawler traverses through the links.Each 
# HTTP request made by the requests library that can be followed
# is an option allowing the Jellyfish to navigate the internet
# whilst observing HTTP connections.
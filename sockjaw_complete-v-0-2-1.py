# now with beustiful soup webparsing for finding his own urls!
# also added random url start from vulnerable liest and some
# forbidden urls for avoiding troule spots.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import uuid
import time
import requests
import aiohttp
import asyncio
from collections import namedtuple
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

# 2Captcha Integration
API_KEY = 'your_2captcha_api'

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Define the Fuzzy Logic Layer
class FuzzyLogicLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(FuzzyLogicLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(input_size, output_size))
        self.bias = nn.Parameter(torch.zeros(output_size))

    def forward(self, x):
        membership_values = torch.sigmoid(torch.mm(x, self.weights) + self.bias)
        return membership_values

# Define the RNN with Fuzzy Logic
class RNNWithFuzzy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNWithFuzzy, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fuzzy_layer = FuzzyLogicLayer(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fuzzy_layer(out[:, -1, :])
        return out

# Define the CNN with Fuzzy Logic
class CNNWithFuzzy(nn.Module):
    def __init__(self):
        super(CNNWithFuzzy, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fuzzy_layer = FuzzyLogicLayer(128, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fuzzy_layer(x)
        x = self.fc2(x)
        return x

# Define the LSTM with Fuzzy Logic
class LSTMWithFuzzy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMWithFuzzy, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fuzzy_layer = FuzzyLogicLayer(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fuzzy_layer(out[:, -1, :])
        return out

# Define the GAN Controller with RNN, CNN, LSTM
class GANWithMultipleNNs(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GANWithMultipleNNs, self).__init__()
        self.rnn = RNNWithFuzzy(input_size, hidden_size, output_size)
        self.cnn = CNNWithFuzzy()
        self.lstm = LSTMWithFuzzy(input_size, hidden_size, output_size)
        self.fc = nn.Linear(output_size * 3, output_size)

    def forward(self, x):
        rnn_out = self.rnn(x)
        cnn_out = self.cnn(x.unsqueeze(1))  # Assuming x needs a channel dimension for CNN
        lstm_out = self.lstm(x)
        combined = torch.cat([rnn_out, cnn_out, lstm_out], dim=1)
        out = self.fc(combined)
        return out

# Define Transition for Reinforcement Learning
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# ProxyPool class with IPv4 and IPv6 proxies
class ProxyPool:
    def __init__(self):
        self.ipv6_proxies = [
            {"proxy": "http://ipv6_proxy1:port", "weight": 5, "failures": 0},
            {"proxy": "http://ipv6_proxy2:port", "weight": 2, "failures": 0},
            {"proxy": "http://ipv6_proxy3:port", "weight": 1, "failures": 0},
        ]
        self.ipv4_proxies = [
            {"proxy": "http://ipv4_proxy1:port", "weight": 5, "failures": 0},
            {"proxy": "http://ipv4_proxy2:port", "weight": 3, "failures": 0},
            {"proxy": "http://ipv4_proxy3:port", "weight": 1, "failures": 0},
        ]

    def get_proxy(self, ipv6=True):
        proxies = self.ipv6_proxies if ipv6 else self.ipv4_proxies
        total_weight = sum(proxy["weight"] for proxy in proxies)
        random_choice = random.uniform(0, total_weight)
        cumulative_weight = 0
        for proxy in proxies:
            cumulative_weight += proxy["weight"]
            if random_choice <= cumulative_weight:
                return proxy["proxy"]
        return None

    def report_failure(self, proxy_url):
        for proxies in [self.ipv6_proxies, self.ipv4_proxies]:
            for proxy in proxies:
                if proxy["proxy"] == proxy_url:
                    proxy["failures"] += 1
                    if proxy["failures"] >= 5:
                        print(f"Removing {proxy_url} due to repeated failures")
                        proxies.remove(proxy)
                    break

    def report_success(self, proxy_url):
        for proxies in [self.ipv6_proxies, self.ipv4_proxies]:
            for proxy in proxies:
                if proxy["proxy"] == proxy_url:
                    proxy["failures"] = 0
                    break

# User-Agent randomization
user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/89.0",
]

# Randomized time delay function
def random_delay(min_delay=1, max_delay=5):
    delay = random.uniform(min_delay, max_delay)
    time.sleep(delay)

# CAPTCHA solving function
def get_recaptcha_response_with_retries_rate_limiting_and_logging(site_key, url, max_retries=5, rate_limit=5):
    last_request_time = 0
    retries = 0

    while retries < max_retries:
        if time.time() - last_request_time < rate_limit:
            wait_time = rate_limit - (time.time() - last_request_time)
            print(f"Rate limit hit. Waiting for {wait_time} seconds...")
            time.sleep(wait_time)

        payload = {
            'key': API_KEY,
            'method': 'userrecaptcha',
            'googlekey': site_key,
            'pageurl': url,
            'json': 1
        }

        try:
            response = requests.post('http://2captcha.com/in.php', data=payload)
            captcha_id = response.json().get('request')

            result_payload = {
                'key': API_KEY,
                'action': 'get',
                'id': captcha_id,
                'json': 1
            }

            while True:
                result = requests.get('http://2captcha.com/res.php', params=result_payload)
                if result.json().get('status') == 1:
                    captcha_text = result.json().get('request')
                    print(f"CAPTCHA solved: {captcha_text}")
                    return captcha_text
                time.sleep(5)
        except Exception as e:
            print(f"Error solving CAPTCHA: {e}")
            retries += 1
            wait_time = 2 ** retries + random.uniform(0, 1)
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

    print("Failed to solve CAPTCHA after multiple retries.")
    return None

# Asynchronous fetch function with proxy and user-agent switching
async def fetch(session, url, proxy_pool, ipv6=True):
    proxy = proxy_pool.get_proxy(ipv6=ipv6)
    headers = {"User-Agent": random.choice(user_agents)}
    random_delay()  # Apply a random delay before each request
    try:
        async with session.get(url, proxy=proxy, headers=headers) as response:
            print(f"Request sent via {proxy}, status code: {response.status}")
            if response.status == 200:

                # Process the response content as needed
                return await response.text()
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}. Retrying with a different proxy...")
        return None

# Define the Jellyfish Agent
class JellyfishAgent:
    def __init__(self, env, gan_brain, ganglion):
        self.env = env
        self.gan_brain = gan_brain
        self.ganglion = ganglion
        self.state = random.choice(env.urls)

    def get_state(self):
        return self.state

    def take_action(self):
        neighbors = self.env.get_neighbors(self.state)
        if not neighbors:
            return self.state, 0
        noise = torch.randn(1, 100).to(device)
        action_probs = self.gan_brain(noise).detach().cpu().numpy().flatten()
        sync_signals = self.ganglion()
        modified_action_probs = action_probs * sync_signals.cpu().numpy()[:len(action_probs)]
        action = random.choices(neighbors, weights=modified_action_probs[:len(neighbors)], k=1)[0]
        self.state = action
        return action, 1

# Define the Network Environment for the Jellyfish AI
class NetworkEnvironment:
    def __init__(self, urls):
        self.urls = urls

    def get_neighbors(self, url):
        return [u for u in self.urls if u != url]  # Return all URLs except the current one

# general settings
gLogFile = "./AAA.txt"
gDEBUG = 2

gBLOCKDestinations = ['\.rapidshare\.com',  
                      '\.blogspot\.com',
                      'www\.google\.',
                      '\.yahoo\.',
                      '\.live\.',
                      '\.gov',
                      ]

gBLOCKURI = ['viagra',
             'cialis',
             'reply',
             'gbook',
             'guestbook',
             'iframe',
             '\/\/ads?\.',
             'http:\/\/[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}',
             'comment'
             ]

# Function to crawl URLs and find new links
def crawl(url, depth=1, max_depth=3):
    if depth > max_depth:
        return []

    visited_urls = set()
    try:
        if url in visited_urls:
            print(f"Already visited: {url}")
            return []

        visited_urls.add(url)

        response = requests.get(url)
        if response.status_code == 200:
            print(f"Crawling: {url}")

            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.find_all('a', href=True)

            discovered_urls = []

            for link in links:
                next_url = link['href']

                if next_url.startswith('http'):
                    discovered_urls.append(next_url)
                    discovered_urls.extend(crawl(next_url, depth + 1, max_depth))

            return discovered_urls

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
                
                # Steer away from URLs ending with ".br"
                if next_url.endswith('.br'):
                    continue
                
                # Steer towards URLs ending with ".com"
                if next_url.endswith('.com'):
                    crawl(next_url, depth + 1, max_depth)

                if next_url.startswith('http'):
                    crawl(next_url, depth + 1, max_depth)
                # You can add more conditions here to steer towards or away from other URLs 
                   
    except Exception as e:
        print(f"Error crawling {url}: {e}")
        return []

# check if destination is blocked
for pattern in gBLOCKDestinations:
   if re.search(pattern, dest_host, re.IGNORECASE):
        self.send_error(HTTPStatus.FORBIDDEN)
        return

# Block certain URIs
for pattern in gBLOCKURI:
   if re.search(pattern, self.path, re.IGNORECASE):
        self.send_error(HTTPStatus.FORBIDDEN)
        return

# Polymorphic code generation function
def generate_polymorphic_code():
    blocks = [
        '''
def check_permission():
    return os.path.exists('/path/to/permission_marker')
''',
        '''
def polymorphic_replicate():
    unique_id = "{unique_id}"
    target_directory = os.getenv('TEMP', '/tmp')
    script_name = 'benign_traveler_v_0_4_0_' + unique_id + '.py'

    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    with open(os.path.join(target_directory, script_name), 'w') as new_script:
        new_script.write(generate_polymorphic_code())
''',
        '''
def should_terminate():
    if random.randint(1, 100) > 85:
        self_destruct()
        return True
    return False
''',
        '''
def self_destruct():
    try:
        os.remove(__file__)
        print("Self-destruct initiated. Script removed.")
    except Exception as e:
        print(f"Self-destruction failed: {e}")
''',
        '''
async def main():
    proxy_pool = ProxyPool()
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _ in range(NUM_THREADS):
            tasks.append(fetch(session, URL, proxy_pool))
        await asyncio.gather(*tasks)
'''
    ]

    random.shuffle(blocks)
    unique_id = str(uuid.uuid4())
    script_body = ''.join(blocks).format(unique_id=unique_id)
    return script_body

# Main function for the polymorphic jellyfish
def main():
    script_code = generate_polymorphic_code()
    exec(script_code)  # Dynamically execute the generated code


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
    # Initialize the environment with initial URLs
    initial_urls = random_choice

    # Start crawling from the initial URLs to discover more
    discovered_urls = []
    for url in initial_urls:
        discovered_urls.extend(crawl(url))

    # Merge discovered URLs with the initial ones
    all_urls = list(set(initial_urls + discovered_urls))

    # Initialize the Network Environment with the discovered URLs
    env = NetworkEnvironment(all_urls)

    # Initialize the GAN brain with RNN, CNN, and LSTM
    input_size = len(all_urls)
    hidden_size = 128
    output_size = len(all_urls)
    gan_brain = GANWithMultipleNNs(input_size, hidden_size, output_size).to(device)

    # Initialize the Ganglion with Cellular Automaton
    ca_shape = (2, 2, 2)
    ca_ruleset = {}
    ganglion = GanglionWithCA(len(all_urls), ca_shape, ca_ruleset).to(device)

    # Initialize the Jellyfish Agent with GAN and Ganglion
    jellyfish = JellyfishAgent(env, gan_brain, ganglion)

    # Train the Jellyfish Agent
    num_epochs = 100
    for epoch in range(num_epochs):
        state = jellyfish.get_state()
        total_reward = 0
        while True:
            action, reward = jellyfish.take_action()
            total_reward += reward
            state = action
            if reward == 1:
                break
        print(f"Epoch {epoch+1}/{num_epochs}, Total Reward: {total_reward}")

if __name__ == "__main__":
    main()

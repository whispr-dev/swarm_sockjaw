import requests
from bs4 import BeautifulSoup
import random
import os
import platform
import socket
import re
import webbrowser    
import subprocess

class InternetJellyfish:
    def __init__(self):
        self.position = "https://www.example.com"  # Starting position
        self.data = []

    def navigate(self):
        # Simulate movement by following a random link on the current page
        response = requests.get(self.position)
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a')
        if links:
            next_link = random.choice(links)['href']
            self.position = next_link
        else:
            print("No links found on this page. Moving randomly.")
            self.position = "https://www.example.com"  # Move to a default page

    def sense(self):
        # Simulate sensing by scraping data from the current page
        response = requests.get(self.position)
        self.data.append(response.text)



<h3>Unanonymity 0.2</h3>
    <b>Tool name</b>   : Unanonymity<br>
    <b>Description</b> : Unanonymity is a PHP script that collects all user data on the web server <br>
side and also on the web client side to show what information the server owner can see <br>
from the user if they want.<br>
    <br>
    <b>Version</b>     : 0.2<br>
    <b>OS</b>          : Firefox 3.0.7 on Windows XP<br>
    <b>Todo</b>        : Javascript portscanner<br>
    <br>
    <br>
    <br>
    <br>
    <h3>Server side information</h3>
    <br>
    <?

      /*
       * Servier side information gathering
       *
       */

      $gRemoteAddr = $_SERVER['HTTP_X_FORWARDED_FOR']?$_SERVER['HTTP_X_FORWARDED_FOR']:$_SERVER['REMOTE_ADDR'];
      $gRemoteHostname = gethostbyaddr($gRemoteAddr);
      $gRemotePort = $_SERVER['REMOTE_PORT'];
      $gUserAgent = $_SERVER['HTTP_USER_AGENT'];
      $gRefererPage = $_SERVER['HTTP_REFERER'];
      $gHTTPAccept = $_SERVER['HTTP_ACCEPT'];
      $gHTTPAcceptLanguage = $_SERVER['HTTP_ACCEPT_LANGUAGE'];
      $gHTTPAcceptEncoding = $_SERVER['HTTP_ACCEPT_ENCODING'];
      $gHTTPAcceptCharset = $_SERVER['HTTP_ACCEPT_CHARSET'];
      $gProxyXForwarded = $_SERVER['HTTP_X_FORWARDED_FOR'];
      $gProxyForwarded = $_SERVER['HTTP_FORWARDED'];
      $gProxyClientIP = $_SERVER['HTTP_CLIENT_IP'];
      $gProxyVia = $_SERVER['HTTP_VIA'];
      $gProxyConnection = $_SERVER['HTTP_PROXY_CONNECTION'];
    ?>
    <p id="coloured1">
    <?
      echo "<b>Your IP address</b> : $gRemoteAddr<br>\n";
      echo "<b>Your hostname</b> : $gRemoteHostname<br>\n";
      echo "<b>Your source port</b> : $gRemotePort<br>\n";
    ?&gt;</p>
      <br>
      <p id="coloured1">
    <?
      echo "<b>Your user agent</b> : $gUserAgent<br>\n";
      echo "<b>Referer page</b> : $gRefererPage<br>\n";
      echo "<b>HTTP Accept</b> : $gHTTPAccept<br>\n";
      echo "<b>HTTP Accept language</b> : $gHTTPAcceptLanguage<br>\n";
      echo "<b>HTTP Accept encoding</b> : $gHTTPAcceptEncoding<br>\n";
      echo "<b>HTTP Accept charset</b> : $gHTTPAcceptCharset<br>\n";
    ?&gt;</p>
      <br>
      <p id="coloured1">
    <?
      echo "<b>Proxy XFORWARDED</b> : $gProxyXForwarded<br>\n";
      echo "<b>Proxy Forwarded</b> : $gProxyForwarded<br>\n";
      echo "<b>Proxy client IP</b> : $ggProxyClientIP<br>\n";
      echo "<b>Proxy via</b> : $gProxyVia<br>\n";
      echo "<b>Proxy Proxy connection</b> : $gProxyConnection<br>\n";
    ?&gt;
    </p>

    <br>
    <br>
    <br>
    <h3>Hardware information</h3>
    <p id="coloured1">
      <script>
        document.write("<b>Browser name</b> : ", navigator.appName, "<br>\n");
        document.write("<b>Browser version</b> : ", navigator.appCodeName, "<br>\n");
        document.write("<b>Platform</b> : ", navigator.platform, "<br>\n");
        document.write("<b>History entries</b> : ", history.length, "<br>\n");
        document.write("<b>Screen size</b> : ", screen.width," x ",screen.height, " x ", screen.colorDepth, " bpp (available for browser: ", window.screen.availWidth, " x ", window.screen.availHeight, ")<br>\n");
      </script>
    </p>
    <br>
    <br>
    <h3>Networking information</h3>
    <p id="coloured1">
      <script>
        if (navigator.javaEnabled())
        {
          addr = java.net.InetAddress.getLocalHost();
          host = addr.getHostName();
          ip = addr.getHostAddress();
          document.write("<b>Localhost address</b> : ", ip, " (hostname: ", host, ")<br>\n");
        }

        if (navigator.javaEnabled())
        {
          host = window.location.host;
          port = window.location.port || 80;
          sock = new java.net.Socket();
          sock.bind(new java.net.InetSocketAddress('0.0.0.0', 0));
          sock.connect(new java.net.InetSocketAddress(host, (!port)?80:port));
          addr = sock.getLocalAddress();
          host = addr.getHostName();
          ip = addr.getHostAddress();
          document.write("<b>Local NIC address</b> : ", ip, " (hostname: ", host, ")<br>\n");
        }
      </script>
    </p>
    <br>
    <br>
    <h3>Installed plugins</h3>
    <p id="coloured1">
      <script>
        if (navigator.plugins.length)
        {
          for (i = 0; i < navigator.plugins.length; i++)
          {
            plugin = navigator.plugins[i];
            document.write(plugin.name, " (", plugin.filename, ")<br>\n");
          }
        }

{
          var color = "";
          var link = "";
          var counter = 0;


          for (counter = 0; counter < websites.length; counter++)
          {
            link = document.createElement("a");
            link.id = "id" + counter;
            link.href = websites[counter];
            link.innerHTML = websites[counter];
            document.body.appendChild(link);
            color = document.defaultView.getComputedStyle(link,null).getPropertyValue("color");
            document.body.removeChild(link);

            if (color == "rgb(0, 0, 255)")
              document.write(websites[counter], "<br>\n");


          } // end visited website loop
        } // end stealHistory method

# Function to parse the PHP output and extract relevant information
def parse_php_output(php_output):
    
    import re

def parse_php_output(php_output):
    # Initialize variables to store parsed data
    user_info = {}
    proxy_info = {}
    hardware_info = {}
    networking_info = {}
    installed_plugins = []
    browser_history = []

    # Regular expressions to match patterns in the PHP output
    user_info_patterns = {
        "IP Address": r"Your IP address\s*:\s*(.*)",
        "Hostname": r"Your hostname\s*:\s*(.*)",
        "Source Port": r"Your source port\s*:\s*(.*)",
        "User Agent": r"Your user agent\s*:\s*(.*)",
        # Add more patterns as needed for other user information
    }

    proxy_info_patterns = {
        # Define patterns for proxy information
    }

    hardware_info_patterns = {
        # Define patterns for hardware information
    }

    networking_info_patterns = {
        # Define patterns for networking information
    }

    # Parse user information
    for key, pattern in user_info_patterns.items():
        match = re.search(pattern, php_output)
        if match:
            user_info[key] = match.group(1)

    # Parse proxy information
                    
                    
                     # Similar parsing logic for other sections...

    return user_info, proxy_info, hardware_info, networking_info, installed_plugins, browser_history

# Example usage
# php_output = """
# Your IP address : 123.456.789.012
# Your hostname : example.com
# Your source port : 12345
# Your user agent : Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.99 Safari/537.36
# """
# user_info, proxy_info, hardware_info, networking_info, installed_plugins, browser_history = parse_php_output(php_output)
# 
# print("User Information:")
# print(user_info)

    # Parse the PHP output here and extract the desired information
    # You may need to use regular expressions or specific parsing logic
    # Return the extracted information in a suitable format

# Main function to simulate the behavior of the jellyfish
def decide(self):
        # Simulate decision making by randomly choosing actions
        actions = ["navigate", "sense"]
        action = random.choice(actions)
        if action == "navigate":
            self.navigate()
        elif action == "sense":
            self.sense()    # Simulate web scraping like a jellyfish "senses", Parse the webpage content to extract relevant information
    extracted_info = parse_php_output(webpage_content)


if __name__ == "__main__":
    jellyfish = InternetJellyfish()
    for _ in range(10):
        jellyfish.decide()
        print("Current Position:", jellyfish.position)
        print("Data Collected:", jellyfish.data[-1][:100])  # Print first 100 characters of the latest data
        print("---")




#    Replace "path/to/your/php/script.php" with the actual path to your PHP script.
#    Implement the parse_php_output function to parse the output of your PHP script and extract the relevant information.
#    Call simulate_jellyfish_behavior to run the entire process.


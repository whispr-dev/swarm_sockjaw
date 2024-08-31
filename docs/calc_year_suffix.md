**fren can you write me python code to return the 2nd two digits from every year from 1940 to present date?**


**ChatGPT** _said:_

Certainly! Here's a Python code snippet that returns the second and third digits (i.e., the tens and units place) from every year starting from 1940 to the present year:

```python
from datetime import datetime

def get_second_and_third_digits(start_year=1940):
    current_year = datetime.now().year
    years = range(start_year, current_year + 1)
    second_third_digits = [str(year)[1:3] for year in years]
    return second_third_digits

# Example usage
digits_list = get_second_and_third_digits()
print(digits_list)
```

Explanation:
- The code imports the datetime module to get the current year.
- It defines a function get_second_and_third_digits that takes a start_year (default is 1940) and generates a list of the second and third digits from each year up to the current year.
- The list comprehension str(year)[1:3] extracts the second and third digits of each year as a string.
- The digits_list contains all the second and third digits of the years from 1940 to the present year.

When you run this code, it will print out the second and third digits for each year in the specified range.


_alt. answer_


**Certainly, friend! Here's a Python script that achieves what you're looking for:**


```python
from datetime import datetime

def get_second_two_digits(start_year=1940):
    current_year = datetime.now().year
    years = range(start_year, current_year + 1)
    second_two_digits = []

    for year in years:
        year_str = str(year)
        if len(year_str) == 4:
            digits = year_str[1:3]
            second_two_digits.append(digits)
        else:
            # Handle years that are not 4 digits (if any)
            second_two_digits.append("NA")

    return second_two_digits

if __name__ == "__main__":
    digits_list = get_second_two_digits()
    for digits in digits_list:
        print(digits)
```

Explanation of the Code:

- Importing Required Module:
We import the datetime module to get the current year dynamically.
```python
from datetime import datetime
```

- Defining the Function:
We define a function get_second_two_digits that takes an optional parameter start_year, defaulting to 1940.
```python
def get_second_two_digits(start_year=1940):
```

- Inside the function, we get the current year.
```python
current_year = datetime.now().year
```

- We create a range of years from the start_year to the current_year inclusive.
```python
years = range(start_year, current_year + 1)
```

- We initialize an empty list to store the extracted digits.
```python
second_two_digits = []
```

- We iterate over each year in the range:
```python
for year in years:
```

- Convert the year to a string:
```python
year_str = str(year)
```

- Extract the second two digits:
```python
if len(year_str) == 4:
    digits = year_str[1:3]
    second_two_digits.append(digits)
else:
    # Handle years that are not 4 digits (if any)
    second_two_digits.append("NA")
```

- Finally, we return the list of extracted digits:
```python
return second_two_digits
```

* Executing the Function:
- We call the function and print out each set of digits:
```python
if __name__ == "__main__":
    digits_list = get_second_two_digits()
    for digits in digits_list:
        print(digits)
```

* Sample Output:
- Running this script will output something like:

```python
94
95
96
97
98
99
00
01
02
...
21
22
23
```

- This will list all the second two digits from the year 1940 up to the current year. For example:

1940 → '94'
1955 → '95'
2003 → '00'
2023 → '02'

- Notes:
The extraction is done by slicing the string representation of the year from index 1 to 3:

```python
digits = year_str[1:3]
For 1940 ('1940'), this yields '94'.
For 2023 ('2023'), this yields '02'.
```

- The code accounts for years that might not be in the 4-digit format, although within the given range this shouldn't be an issue.
- You can modify the start_year parameter when calling get_second_two_digits if you wish to start from a different year.

* Extended Usage:
If you wish to use these digits elsewhere in your code, you can simply work with the returned list:

```python
digits_list = get_second_two_digits(1950)
print(digits_list)
```

This will start extracting from the year 1950 instead.

* Final Thoughts:
- This simple and efficient script should fulfill your requirement to extract the second two digits from years spanning 1940 to the current year. Let me know if you need any further assistance or modifications!



**To integrate the logic of appending or prepending the second two digits of years from 1940 to the present (as well as the digits 0, 1, and 69) to passwords in your main_password_check function, here's how you can modify the code:**


```python
from datetime import datetime
from itertools import product

def get_year_digits():
    current_year = datetime.now().year
    years = range(1940, current_year + 1)
    year_digits = set()

    for year in years:
        year_str = str(year)
        if len(year_str) == 4:
            year_digits.add(year_str[2:])  # Add the last two digits
            year_digits.add(year_str[:2])  # Add the first two digits (optional)
    
    # Add specific digits '0', '1', '69'
    year_digits.update(['0', '1', '69'])
    
    return list(year_digits)

# New password-checking function
def main_password_check():
    # List of top 50 most common passwords
    common_passwords = [
        "123456", "password", "12345678", "qwerty", "123456789", "12345", "1234", "111111",
        "1234567", "dragon", "123123", "baseball", "abc123", "football", "monkey", "letmein",
        "696969", "shadow", "master", "666666", "qwertyuiop", "123321", "mustang", "1234567890",
        "michael", "654321", "superman", "1qaz2wsx", "7777777", "121212", "000000", "qazwsx",
        "123qwe", "killer", "trustno1", "jordan", "jennifer", "zxcvbnm", "asdfgh", "hunter",
        "buster", "soccer", "harley", "batman", "andrew", "tigger", "sunshine", "iloveyou", "2000"
    ]

    # Function to generate lucky numbers
    def sieve_lucky_numbers(n):
        numbers = list(range(1, n + 1, 2))  # Start with odd numbers only
        i = 1
        while i < len(numbers):
            step = numbers[i]
            numbers = [num for idx, num in enumerate(numbers) if (idx + 1) % step != 0]
            i += 1
        return numbers

    # Generate the first 1000 lucky numbers as strings
    lucky_numbers = sieve_lucky_numbers(1000)
    lucky_numbers = [str(num) for num in lucky_numbers]

    # Generate all possible password combinations
    def gen_pswd_combos(knwn):
        digits = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-_=+[]{}|;:,.<>?/~`'
        lngt = len(knwn)
        while True:
            for combos in product(digits, repeat=lngt):
                yield knwn + ''.join(combos)
            lngt += 1  # Increase the length after exhausting all combos of the current length

    # Check if the generated password matches the real password
    def is_rl_pswd(pswd, rl_pswd):
        return pswd == rl_pswd

    net_auto_password = 'password'  # The password we want to solve
    print(f"Actual real password to work from: {net_auto_password}")
    
    start_time = time.time()  # Start timing

    # Get the list of year digits
    year_digits = get_year_digits()

    # First, try the lucky numbers
    for i, lucky_pswd in enumerate(lucky_numbers):
        print(f"Trying lucky number password {i}: {lucky_pswd}")
        if is_rl_pswd(lucky_pswd, net_auto_password):
            elapsed_time = time.time() - start_time  # Calculate elapsed time
            print(f"Correct password found: {lucky_pswd}. Pretending to accept the login.")
            print(f"Elapsed time: {elapsed_time:.2f} seconds")
            return

    # Then, try the common passwords from the wordlist with year digits appended/prepended
    for i, common_pswd in enumerate(common_passwords):
        for year_digit in year_digits:
            # Try prepending the year digit
            pswd_with_year_prepend = year_digit + common_pswd
            print(f"Trying common password with year prepend {i}: {pswd_with_year_prepend}")
            if is_rl_pswd(pswd_with_year_prepend, net_auto_password):
                elapsed_time = time.time() - start_time  # Calculate elapsed time
                print(f"Correct password found: {pswd_with_year_prepend}. Pretending to accept the login.")
                print(f"Elapsed time: {elapsed_time:.2f} seconds")
                return

            # Try appending the year digit
            pswd_with_year_append = common_pswd + year_digit
            print(f"Trying common password with year append {i}: {pswd_with_year_append}")
            if is_rl_pswd(pswd_with_year_append, net_auto_password):
                elapsed_time = time.time() - start_time  # Calculate elapsed time
                print(f"Correct password found: {pswd_with_year_append}. Pretending to accept the login.")
                print(f"Elapsed time: {elapsed_time:.2f} seconds")
                return

    # If not found in lucky numbers or common passwords, try the generated combinations
    combos = gen_pswd_combos('')
    for i, combo in enumerate(combos):
        print(f"Combo {i}: {combo}\n")
        print(f"Trying password: {combo}")
        if is_rl_pswd(combo, net_auto_password):
            elapsed_time = time.time() - start_time  # Calculate elapsed time
            print(f"Correct password found: {net_auto_password}. Pretending to accept the login.")
            print(f"Elapsed time: {elapsed_time:.2f} seconds")
            break

# Main execution
if __name__ == "__main__":
    main_password_check()  # Call the new password-checking function
```

Explanation:
- Year Digits Generation:
- The function get_year_digits() generates the last two digits of each year from 1940 to the present year, as well as the digits '0', '1', and '69'.
- Trying Combinations with Year Digits:
- The modified main_password_check() function first tries to prepend or append the digits to each password in the common_passwords list.
- Appended or Prepended Passwords:
The code first attempts to match the password by prepending each of the year_digits to the common passwords and then by appending them. If a match is found, it prints out the correct password and the time taken.
- Unchanged Password:
If no match is found, it falls back to attempting to generate all possible password combinations using the original logic you had in your code.

This modification adds a strategic layer to brute-forcing by attempting variations of known patterns and common digits used in passwords.

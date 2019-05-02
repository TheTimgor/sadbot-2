import requests
import re


def correct(sentence):
    query = re.sub('\s', '+', sentence)
    response = requests.get(f"https://montanaflynn-spellcheck.p.rapidapi.com/check/?text={query}",
                            headers={
                              "X-RapidAPI-Host": "montanaflynn-spellcheck.p.rapidapi.com",
                              "X-RapidAPI-Key": "5cfb972b2dmsh6aee520c478ecc6p17e701jsne5c8623eecb4",
                              "Content-Type": "application/x-www-form-urlencoded"
                            }
    )
    return response.json()['suggestion']

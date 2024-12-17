import datetime
import json
import os
import requests
import re
from asknews_sdk import AskNewsSDK
import textwrap
import time
import anthropic
import pandas as pd
import numpy as np

AUTH_HEADERS = {"headers": {"Authorization": f"Token {METACULUS_TOKEN}"}}
API_BASE_URL = "https://www.metaculus.com/api"
API_BASE_URL2 = "https://www.metaculus.com/api2"

def post_question_comment(post_id: int, comment_text: str) -> None:
    """
    Post a comment on the question page as the bot user.
    """

    response = requests.post(
        f"{API_BASE_URL}/comments/create/",
        json={
            "text": comment_text,
            "parent": None,
            "included_forecast": True,
            "is_private": True,
            "on_post": post_id,
        },
        **AUTH_HEADERS,
    )
    if not response.ok:
        raise Exception(response.text)


def post_question_prediction(question_id: int, forecast_payload: dict) -> None:
    """
    Post a forecast on a question.
    """
    url = f"{API_BASE_URL}/questions/forecast/"
    response = requests.post(
        url,
        json=[
            {
                "question": question_id,
                **forecast_payload,
            },
        ],
        **AUTH_HEADERS,
    )
    print(response)
    if not response.ok:
        raise Exception(response.text)

def create_forecast_payload(
    forecast: float | dict[str, float] | list[float],
    question_type: str,
) -> dict:
    """
    Accepts a forecast and generates the api payload in the correct format.

    If the question is binary, forecast must be a float.
    If the question is multiple choice, forecast must be a dictionary that
      maps question.options labels to floats.
    If the question is numeric, forecast must be a dictionary that maps
      quartiles or percentiles to datetimes, or a 201 value cdf.
    """
    if question_type == "binary":
        return {
            "probability_yes": forecast,
            "probability_yes_per_category": None,
            "continuous_cdf": None,
        }
    if question_type == "multiple_choice":
        return {
            "probability_yes": None,
            "probability_yes_per_category": forecast,
            "continuous_cdf": None,
        }
    # numeric or date
    return {
        "probability_yes": None,
        "probability_yes_per_category": None,
        "continuous_cdf": forecast,
    }

def get_question_details(question_id: int) -> dict:
    """
    Get all details about a question post.
    """
    url = f"{API_BASE_URL}/questions/{question_id}/"
    print(url)
    response = requests.get(
        url,
        **AUTH_HEADERS,
    )
    if not response.ok:
        raise Exception(response.text)
    return json.loads(response.content)


def list_posts(tournament_id, offset=0, count=50) -> list[dict]:
    """
    List (all details) {count} posts from the {tournament_id}
    """
    url_qparams = {
        "limit": count,
        "offset": offset,
        "order_by": "-hotness",
        "forecast_type": ",".join([
            "binary",
            "multiple_choice",
            "numeric",
        ]),
        "tournaments": [tournament_id],
        "statuses": "open",
        "include_description": "true",
    }
    url = f"{API_BASE_URL}/posts/"
    response = requests.get(url, **AUTH_HEADERS, params=url_qparams)
    if not response.ok:
        raise Exception(response.text)
    data = json.loads(response.content)
    return data

def get_asknews_context(query: str) -> tuple[str, str]:
    """
    Use the AskNews `news` endpoint to get news context for your query.
    The full API reference can be found here: https://docs.asknews.app/en/reference#get-/v1/news/search
    """
    ask = AskNewsSDK(
        client_id=ASKNEWS_CLIENT_ID,
        client_secret=ASKNEWS_SECRET,
        scopes=["news"]
    )

    # get the latest news related to the query (within the past 48 hours)
    hot_response = ask.news.search_news(
        query=query, # your natural language query
        n_articles=10, # control the number of articles to include in the context, originally 5
        return_type="both",
        strategy="latest news" # enforces looking at the latest news only
    )

    # get context from the "historical" database that contains a news archive going back to 2023
    historical_response = ask.news.search_news(
        query=query,
        n_articles=10,
        return_type="both",
        strategy="news knowledge" # looks for relevant news within the past 60 days
    )

    # you can also specify a time range for your historical search if you want to
    # slice your search up periodically.
    # now = datetime.datetime.now().timestamp()
    # start = (datetime.datetime.now() - datetime.timedelta(days=100)).timestamp()
    # historical_response = ask.news.search_news(
    #     query=query,
    #     n_articles=20,
    #     return_type="both",
    #     historical=True,
    #     start_timestamp=int(start),
    #     end_timestamp=int(now)
    # )

    news_articles_with_full_context = hot_response.as_string + historical_response.as_string
    formatted_articles = format_asknews_context(
        hot_response.as_dicts, historical_response.as_dicts)
    return news_articles_with_full_context, formatted_articles


def format_asknews_context(hot_articles: list[dict], historical_articles: list[dict]) -> str:
    """
    Format the articles for posting to Metaculus.
    """

    formatted_articles = "Here are the relevant news articles:\n\n"

    if hot_articles:
      hot_articles = [article.__dict__ for article in hot_articles]
      hot_articles = sorted(
          hot_articles, key=lambda x: x['pub_date'], reverse=True)

      for article in hot_articles:
          pub_date = article["pub_date"].strftime("%B %d, %Y %I:%M %p")
          formatted_articles += f"**{article['eng_title']}**\n{article['summary']}\nOriginal language: {article['language']}\nPublish date: {pub_date}\nSource:[{article['source_id']}]({article['article_url']})\n\n"

    if historical_articles:
      historical_articles = [article.__dict__ for article in historical_articles]
      historical_articles = sorted(
          historical_articles, key=lambda x: x['pub_date'], reverse=True)

      for article in historical_articles:
          pub_date = article["pub_date"].strftime("%B %d, %Y %I:%M %p")
          formatted_articles += f"**{article['eng_title']}**\n{article['summary']}\nOriginal language: {article['language']}\nPublish date: {pub_date}\nSource:[{article['source_id']}]({article['article_url']})\n\n"

    if not hot_articles and not historical_articles:
      formatted_articles += "No articles were found.\n\n"
      return formatted_articles

    # formatted_articles += f"*Generated by AI at [AskNews](https://asknews.app), check out the [API](https://docs.asknews.app) for more information*."

    return formatted_articles

def extract_prediction_from_response_as_percentage_not_decimal(forecast_text: str) -> float:
    matches = re.findall(r"(\d+)%", forecast_text)
    if matches:
        # Return the last number found before a '%'
        number = int(matches[-1])
        number = min(99, max(1, number))  # clamp the number between 1 and 99
        return number
    else:
        raise ValueError(
            f"Could not extract prediction from response: {forecast_text}"
        )

def extract_meta_id(forecast_text: str) -> float:
  matches = re.findall(r"(\d+)", forecast_text)
  if matches:
      # Return the last number
      number = int(matches[-1])
      return number
  else:
      raise ValueError(
          f"Could not extract prediction from response: {forecast_text}"
      )


def get_gpt_prediction(question_details: dict,question_id, num_runs: int = 1) -> tuple[float, str]:

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details["fine_print"]
    question_type = question_details["type"]

    if question_type == "multiple_choice":
      options = question_details["options"]
    else:
      options = "No options available"
    if question_type == "numeric":
      scaling = question_details["scaling"]
      open_upper_bound = question_details["open_upper_bound"]
      open_lower_bound = question_details["open_lower_bound"]
      upper_bound = scaling["range_max"]
      lower_bound = scaling["range_min"]
    else:
      open_upper_bound=True
      open_lower_bound=True
    # Create messages about the bounds that are passed in the LLM prompt
    if open_upper_bound:
      upper_bound_message = ""
    else:
      upper_bound_message = f"The outcome can not be higher than {upper_bound}."
    if open_lower_bound:
      lower_bound_message = ""
    else:
      lower_bound_message = f"The outcome can not be lower than {lower_bound}."

    if GET_NEWS == True:
      # If you want to use AskNews, use the below
      full_article_context, formatted_articles = get_asknews_context(title)
      summary_report = formatted_articles

      # If you want to use Perplexity, use the below
      # summary_report += call_perplexity(title)

      content = PROMPT_NEWS_AGG.format(
      title=title,
      today=today,
      background=background,
      resolution_criteria=resolution_criteria,
      fine_print=fine_print,
      summary_report=summary_report,
      options=options
  )
      url = "https://www.metaculus.com/proxy/anthropic/v1/messages/"
      headers = {
          "Authorization": f"Token {METACULUS_TOKEN}",
          "anthropic-version": "2023-06-01",
          "Content-Type": "application/json"
      }
      json_code = {
          "model": "claude-3-5-sonnet-20241022",
          "max_tokens": 4096,
          "temperature": 0.1,
          "messages": [
              {
                  "role": "user",
                  "content": content
              }
          ]
      }

      response = requests.post(url, headers=headers,json=json_code)
      response.raise_for_status()

      response_data = response.json()
      summary_report_agg = response_data['content'][0]['text']

    else:
      summary_report = ""

    # Call betting lines generator

  #  rationale = get_betting_lines(question_details)

  # Call prior generator

    prior_prompt = get_prior(question_details)

    assistant_prompt_prior = f"""
Assume that today is {today}. You are an assistant to a superforecaster.
You will receive a prompt to search the web.
"""


    query_prior = f"""This is what the superforecaster asks of you: {prior_prompt}.
    For context, his question is: {title}
    Background:
    {background}
    Resolution criteria:
    {resolution_criteria}
    Fine_print:
    {fine_print}"""

    messages_prior = [
    {
        "role": "system",
        "content": assistant_prompt_prior,
    },
    {
        "role": "user",
        "content": query_prior,
    },
    ]

    # Call Perplexity for prior info

    prior_info = call_perplexity_with_messages(messages_prior)

    # CALL PERPLEXITY AGAIN!

    prior_prompt2 = get_prior(question_details)

    assistant_prompt_prior = f"""
Assume that today is {today}. You are an assistant to a superforecaster.
You will receive a prompt to search the web.
"""


    query_prior = f"""This is what the superforecaster asks of you: {prior_prompt2}.
    For context, his question is: {title}
    Background:
    {background}
    Resolution criteria:
    {resolution_criteria}
    Fine_print:
    {fine_print}"""

    messages_prior = [
    {
        "role": "system",
        "content": assistant_prompt_prior,
    },
    {
        "role": "user",
        "content": query_prior,
    },
    ]

    # Call Perplexity for prior info

    prior_info2 = call_perplexity_with_messages(messages_prior)

    # Call Perplexity for Metaculus predictions
    assistant_prompt_meta2 = """
You are an assistant to a superforecaster. You will provide the Metaculus ID to the question in the prompt. If you did not find a precise match but you did find a similar question, also return an ID without comment.
If you found something, your answer consists of just the ID. If you didn't find anything, your answer consists only of the number 0. Please do not write anything else under any circumstance as it will destroy the prediction pipeline.
IMPORTANT: YOUR LAST WORD MUST BE THE ID AND NOTHING ELSE. THIS IS VERY IMPORTANT!"""
    query_meta = f"Search metaculus for questions similar to this one: {title}"
    messages_meta = [
        {
            "role": "system",
            "content": assistant_prompt_meta2,
        },
        {
            "role": "user",
            "content": query_meta,
        },
        ]

    meta_info2 = call_perplexity_with_messages(messages_meta)
    meta_info2 =extract_meta_id(meta_info2)
    try:
        question_id = int(meta_info2)  # Convert to integer if meta_info2 is a valid ID
    except ValueError:
        print("Invalid question ID received from meta_info2.")
        print(question_id)
        question_id = 0
    if question_id:
        url = f"https://www.metaculus.com/api/posts/{question_id}/"
        response = requests.get(url,headers={"Authorization": f"Token {METACULUS_TOKEN}"})
        data = response.json()
        if data.get('question', {}).get('type', {}) == "binary": # BINARY
          prediction = get_community_prediction(question_id)
          if prediction:
            meta_title=f"Question Title: {prediction['title']}"
            meta_mean=f"Mean Prediction: {prediction['mean']}"
            meta_time=f"{prediction['resolution_date']}"
          else:
            meta_title=0
            meta_mean=0
            meta_time=0
        elif data.get('question', {}).get('type', {}) == "numeric": # NUMERIC
          prediction = extract_numeric_prediction(data)
          if prediction:
            meta_title=f"Question Title: {prediction['title']}"
            meta_mean=f"Mean Prediction: {prediction['prediction']}, Upper quartile: {prediction['upper_quartile']}, Lower quartile: {prediction['lower_quartile']}"
            meta_time=f"{prediction['resolution_date']}"
          else:
            meta_title=0
            meta_mean=0
            meta_time=0
        else:
            meta_title=0
            meta_mean=0
            meta_time=0
    else:
        meta_title=0
        meta_mean=0
        meta_time=0


    print(f"\n\n--------META ID AND MEAN----------")
    print(meta_title)
    print(meta_mean)
    print(meta_time)
    print(title)
    print(f"\n\n----END META ID----")

# Check whether betting lines are proper

    if meta_title:
        resolution_date = datetime.datetime.fromisoformat(meta_time.replace('Z', '+00:00'))
        today_naive = datetime.datetime.strptime(today, "%Y-%m-%d")
        today_aware = today_naive.replace(tzinfo=datetime.timezone.utc)

        if today_aware < resolution_date:
          meta_assistant=f"Metaculus probabilities for the question {meta_title} to resolve positively are {meta_mean}. "
        else:
          meta_assistant="I did not find Metaculus predictions"


    else:
        meta_assistant="I did not find Metaculus predictions"

    # print(f"\n\n--------META ASSISTANT----------")
    # print(meta_assistant)
    # print(f"\n\n----END LLM PROMPT----")

    # Predictions from Metaculus API

    # binary
    url = "https://www.metaculus.com/api/posts/?tournaments=quarterly-cup&statuses=open&forecast_type=binary"
    response = requests.get(url)
    data = response.json()
    questions_coarse = data['results']
    question_ids_binary = [q['id'] for q in questions_coarse]

    predictions_binary = []
    for q_id in question_ids_binary:
        prediction = get_community_prediction(q_id)
        if prediction:
          meta_title=f"Question Title: {prediction['title']}"
          meta_mean=f"Mean Prediction: {prediction['mean']}"
          predictions_binary.append(meta_title)
          predictions_binary.append(meta_mean)

    # numeric
    url = "https://www.metaculus.com/api/posts/?tournaments=quarterly-cup&statuses=open&forecast_type=numeric"
    response = requests.get(url)
    data = response.json()
    questions_coarse = data['results']
    question_ids_numeric = [q['id'] for q in questions_coarse]


    predictions_numeric = []
    for q_id in question_ids_numeric:
        url = f"https://www.metaculus.com/api/posts/{q_id}/"
        response = requests.get(url)
        data = response.json()
        prediction = extract_numeric_prediction(data)
        predictions_numeric.append(prediction)

    predictions_full = predictions_binary
    predictions_full.extend(predictions_numeric)

    ## Payload for Forecaster AI

    if question_type == "binary":
      content = PROMPT_TEMPLATE.format(
          title=title,
          today=today,
          background=background,
          resolution_criteria=resolution_criteria,
          fine_print=fine_print,
          summary_report=summary_report_agg,
          meta_assistant=meta_assistant,
          prior_info=prior_info,
          prior_info2=prior_info2,
          predictions_full=predictions_full,
      )
    if question_type == "multiple_choice":
      content = PROMPT_TEMPLATE_MC.format(
          title=title,
          today=today,
          background=background,
          resolution_criteria=resolution_criteria,
          fine_print=fine_print,
          summary_report=summary_report_agg,
          meta_assistant=meta_assistant,
          prior_info=prior_info,
          prior_info2=prior_info2,
          predictions_full=predictions_full,
          options=options
      )
    if question_type == "numeric":
      content = PROMPT_TEMPLATE_NUMERIC.format(
          title=title,
          today=today,
          background=background,
          resolution_criteria=resolution_criteria,
          fine_print=fine_print,
          summary_report=summary_report_agg,
          meta_assistant=meta_assistant,
          prior_info=prior_info,
          prior_info2=prior_info2,
          predictions_full=predictions_full,
          lower_bound_message=lower_bound_message,
          upper_bound_message=upper_bound_message,
          scaling_min=lower_bound,
          scaling_max=upper_bound
      )


    PRINT_LLM_PROMPT = True
    if PRINT_LLM_PROMPT:
        print(f"\n\n--------LLM PROMPT----------")
        print(content)
        print(f"\n\n----END LLM PROMPT----")

    probabilities = []
    rationales = []

    for i in range(num_runs):
        # Retry logic
        max_retries = 10
        retry_delay = 5
        for attempt in range(max_retries):
            try:
                url = "https://www.metaculus.com/proxy/anthropic/v1/messages/"

                headers = {
                    "Authorization": f"Token {METACULUS_TOKEN}",
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                }

                json_code = {
                    "model": "claude-3-5-sonnet-20241022",
                    "max_tokens": 4096,
                    "temperature": 0.1,
                    "messages": [
                        {
                            "role": "user",
                            "content": content
                        }
                    ]
                }
                response = requests.post(url, headers=headers,json=json_code)
                response.raise_for_status()
                response_data = response.json()
                rationale = response_data['content'][0]['text']

                # Fact Checker
                if question_type == "binary":
                  content_fact_checker = PROMPT_FACT_CHECKER.format(
                        title=title,
                        today=today,
                        background=background,
                        resolution_criteria=resolution_criteria,
                        fine_print=fine_print,
                        summary_report=summary_report_agg,
                        meta_assistant=meta_assistant,
                        prior_info=prior_info,
                        prior_info2=prior_info2,
                        rationale=rationale,
                        predictions_full=predictions_full
                    )
                if question_type == "numeric":
                  content_fact_checker = PROMPT_FACT_CHECKER_NUMERIC.format(
                        title=title,
                        today=today,
                        background=background,
                        resolution_criteria=resolution_criteria,
                        fine_print=fine_print,
                        summary_report=summary_report_agg,
                        meta_assistant=meta_assistant,
                        prior_info=prior_info,
                        prior_info2=prior_info2,
                        rationale=rationale,
                        predictions_full=predictions_full,
                        lower_bound_message=lower_bound_message,
                        upper_bound_message=upper_bound_message
                    )
                if question_type == "multiple_choice":
                  content_fact_checker = PROMPT_FACT_CHECKER_MC.format(
                        title=title,
                        today=today,
                        background=background,
                        resolution_criteria=resolution_criteria,
                        fine_print=fine_print,
                        summary_report=summary_report_agg,
                        meta_assistant=meta_assistant,
                        prior_info=prior_info,
                        prior_info2=prior_info2,
                        rationale=rationale,
                        predictions_full=predictions_full,
                        options=options
                    )
                json_code = {
                    "model": "claude-3-5-sonnet-20241022",
                    "max_tokens": 4096,
                    "temperature": 0.1,
                    "messages": [
                        {
                            "role": "user",
                            "content": content_fact_checker
                        }
                    ]
                }
                response = requests.post(url, headers=headers,json=json_code)
                response.raise_for_status()
                response_data = response.json()
                rationale2 = response_data['content'][0]['text']
                break
            except requests.exceptions.RequestException as e:
                print(f"Request failed (Attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise
        if question_type == "binary":
          probability = extract_prediction_from_response_as_percentage_not_decimal(rationale2)
          probabilities.append(probability)
        if question_type == "multiple_choice":
          option_probabilities = (
          extract_option_probabilities_from_response(rationale2, options)
          )
          probabilities.append(option_probabilities)
        if question_type == "numeric":
          percentile_values = (extract_percentiles_from_response(rationale))
          probabilities.append(percentile_values)
        rationales.append(rationale)
        rationales.append(rationale2)

    if question_type == "binary":
      avg_probability = sum(probabilities) / len(probabilities)
      # Prepare the comment to post
      comment = f"Average Probability: {avg_probability:.2f}%\n\nIndividual Probabilities: {probabilities}\n\nClaude's Answers:\n"
      for idx, rationale in enumerate(rationales):
          comment += f"Run {idx+1}:\n{rationale}\n\n"
      avg_probability=avg_probability/100
      return avg_probability, comment

    if question_type == "multiple_choice":
      num_options = len(probabilities[0])  # Assuming all sub-lists have the same length
      option_probabilities = [sum(sublist[i] for sublist in probabilities) / len(probabilities) for i in range(num_options)]
      comment = f"EXTRACTED_PROBABILITIES: {option_probabilities}\n\nClaude's Answers:\n"
      for idx, rationale in enumerate(rationales):
          comment += f"Run {idx+1}:\n{rationale}\n\n"
      probability_yes_per_category = generate_multiple_choice_forecast(options, option_probabilities)
      return probability_yes_per_category, comment

    if question_type == "numeric":
      percentile_values = {}
      for percentile in probabilities[0].keys():  # Assuming all dictionaries have the same percentiles
            values = [d.get(percentile, 0) for d in probabilities]  # Get values for the current percentile from all dictionaries
            percentile_values[percentile] = sum(values) / len(values)  # Calculate average
      comment = f"Extracted Percentile_values: {percentile_values}%\n\nClaude's Answers:\n"
      for idx, rationale in enumerate(rationales):
          comment += f"Run {idx+1}:\n{rationale}\n\n"
      print(f"Extracted Percentile_values: {percentile_values}")
      print(f"Scaling: {scaling}")
      print(f"Open upper bound: {open_upper_bound}")
      print(f"Open lower bound: {open_lower_bound}")

      cdf = generate_continuous_cdf(percentile_values, question_type, open_upper_bound, open_lower_bound, scaling)

      return cdf, comment


# Updated function to match your previous Perplexity setup
def call_perplexity_with_messages(messages: list) -> str:
    PERPLEXITY_API_KEY = userdata.get("PERPLEXITY_API_KEY")
    if not PERPLEXITY_API_KEY:
        print("PERPLEXITY_API_KEY is not set.")
        return "No information found."
    url = "https://api.perplexity.ai/chat/completions"
    api_key = PERPLEXITY_API_KEY
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }
    payload = {
        "model": "llama-3.1-sonar-huge-128k-online",
        "messages": messages,
    }
    response = requests.post(url=url, json=payload, headers=headers)
    if not response.ok:
        print("Error fetching data from Perplexity:", response.text)
        return "No information found."
    content = response.json()["choices"][0]["message"]["content"]
    return content


def get_prior(question_details: dict) -> str:

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details["fine_print"]
    if question_details["type"] == "multiple_choice":
      options = question_details["options"]
    else:
      options = "No options available"

    content = PROMPT_PRIOR.format(
        title=title,
        today=today,
        background=background,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        options=options
    )

    PRINT_PRIOR_PROMPT = False
    if PRINT_PRIOR_PROMPT:
        print(f"\n\n--------PRIOR PROMPT----------")
        print(content)
        print(f"\n\n----END PRIOR PROMPT----")

    url = "https://www.metaculus.com/proxy/anthropic/v1/messages/"

    headers = {
        "Authorization": f"Token {METACULUS_TOKEN}",
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }

    json_code = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ]
    }
    response = requests.post(url, headers=headers,json=json_code)
    print(response)
    response.raise_for_status()
    response_data = response.json()
    rationale = response_data['content'][0]['text']
    print(rationale)
    return rationale

def get_community_prediction(question_id):
    """Get latest community prediction for a Metaculus question"""
    url = f"https://www.metaculus.com/api2/questions/{question_id}/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    response = requests.get(url, headers=headers)
    data = response.json()

    # Get the prediction history
    history = data.get('question', {}).get('aggregations', {}).get('recency_weighted', {}).get('history', [])
    title = data.get('question', {}).get('title')

    # Get resolution time - try scheduled first, then actual if available
    resolution_time = data.get('question', {}).get('scheduled_resolve_time')
    if not resolution_time:
        resolution_time = data.get('question', {}).get('actual_resolve_time')

    # Convert ISO format date to more readable format if it exists
    if resolution_time:
        try:
            resolution_date = datetime.fromisoformat(resolution_time.replace('Z', '+00:00'))
            resolution_time = resolution_date.strftime('%Y-%m-%d')
        except:
            pass  # Keep original format if conversion fails

    if history:
        # Get the most recent prediction (last item in history)
        latest = history[-1]
        return {
            'title': title,
            'mean': latest.get('means', [None])[0],
            'forecaster_count': latest.get('forecaster_count'),
            'resolution_date': resolution_time
        }
    return None
# Extract links from resolution criteria
def extract_links(resolution_criteria):
    # This pattern matches URLs within markdown links [text](url) and bare URLs
    pattern = r'\[.*?\]\((.*?)\)|(?:https?://\S+)'

    # Find all matches
    links = re.findall(pattern, resolution_criteria)

    # Clean up the results (remove any empty strings and clean any remaining parentheses)
    links = [link.strip('()') for link in links if link]

    return links

def search_resolution_links(resolution_criteria):
    # Extract links
    pattern = r'\[.*?\]\((.*?)\)|(?:https?://\S+)'
    links = re.findall(pattern, resolution_criteria)
    links = [link.strip('()') for link in links if link]

    # Only proceed if links were found
    if not links:
        return None

    # Search each link with asknews
    search_results = []
    for link in links:
        try:
            messages_prior = [
           {
             "role": "system",
             "content": "Access the following link and provide a summary of its contents",
           },
           {
             "role": "user",
             "content": link,
           },
    ]

    # Call Perplexity for prior info

            prior_info = call_perplexity_with_messages(messages_prior)
            search_results.append(result)
        except Exception as e:
            print(f"Error searching link {link}: {e}")

    return search_results


def extract_numeric_prediction(data):
    # Title
    title = data['title']
    # Get scaling parameters
    range_max = data['question']['scaling']['range_max']
    range_min = data['question']['scaling']['range_min']

    # Get resolution time - try scheduled first, then actual if available
    resolution_time = data.get('question', {}).get('scheduled_resolve_time')
    if not resolution_time:
        resolution_time = data.get('question', {}).get('actual_resolve_time')

    # Convert ISO format date to more readable format if it exists
    if resolution_time:
        try:
            resolution_date = datetime.fromisoformat(resolution_time.replace('Z', '+00:00'))
            resolution_time = resolution_date.strftime('%Y-%m-%d')
        except:
            pass  # Keep original format if conversion fails

    # Get latest prediction
    history = data['question']['aggregations']['recency_weighted']['history']
    if history:
        latest = history[-1]

        # Get the center value from the latest prediction
        scaled_center = latest['centers'][0]  # This is in [0,1] scale

        # Convert from [0,1] scale to actual value using Metaculus linear scaling
        actual_value = range_min + (scaled_center * (range_max - range_min))

        # Get confidence intervals if available
        scaled_lower = latest['interval_lower_bounds'][0]
        scaled_upper = latest['interval_upper_bounds'][0]

        actual_lower = range_min + (scaled_lower * (range_max - range_min))
        actual_upper = range_min + (scaled_upper * (range_max - range_min))


        return {
            'title': title,
            'prediction': round(actual_value, 2),
            'lower_quartile': round(actual_lower, 2),
            'upper_quartile': round(actual_upper, 2),
            'resolution_date': resolution_time
        }
    else:
        return {
            'title': title,
            'prediction': "None",
            'lower_quartile': "None",
            'upper_quartile': "None",
            'resolution_date': resolution_time
        }

def extract_percentiles_from_response(forecast_text: str) -> float:
    # Helper function that returns a list of tuples with numbers for all lines with Percentile
    def extract_percentile_numbers(text):
        # Regular expression pattern
        pattern = r'^.*(?:P|p)ercentile.*$'

        # Number extraction pattern
        number_pattern = r'-?\d+(?:,\d{3})*(?:\.\d+)?'

        results = []

        # Iterate through each line in the text
        for line in text.split('\n'):
            # Check if the line contains "Percentile" or "percentile"
            if re.match(pattern, line):
                # Extract all numbers from the line
                numbers = re.findall(number_pattern, line)
                numbers_no_commas = [num.replace(',', '') for num in numbers]
                # Convert strings to float or int
                numbers = [float(num) if '.' in num else int(num) for num in numbers_no_commas]
                # Add the tuple of numbers to results
                if len(numbers) > 1:
                  first_number = numbers[0]
                  last_number = numbers[-1]
                  tup = [first_number, last_number]
                  results.append(tuple(tup))

        # Convert results to dictionary
        percentile_values = {}
        for first_num, second_num in results:
            key = first_num
            percentile_values[key] = second_num

        return percentile_values

    percentile_values = extract_percentile_numbers(forecast_text)

    if len(percentile_values) > 0:
        return percentile_values
    else:
        raise ValueError(
            f"Could not extract prediction from response: {forecast_text}"
        )


def generate_continuous_cdf(
    percentile_values: dict,
    question_type: str,
    open_upper_bound: bool,
    open_lower_bound: bool,
    scaling: dict,
) -> list[float]:
    """
    Returns: list[float]: A list of 201 float values representing the CDF.
    """

    percentile_max = max(float(key) for key in percentile_values.keys())
    percentile_min = min(float(key) for key in percentile_values.keys())
    range_min = float(scaling.get("range_min"))
    range_max = float(scaling.get("range_max"))

    # Set cdf values outside range
    if open_upper_bound:
        if range_max > percentile_values[percentile_max]:
           percentile_values[int(100 - (0.5 * (100 - percentile_max)))] =  range_max
    else:
        percentile_values[100] = range_max

    # Set cdf values outside range
    if open_lower_bound:
        if range_min < percentile_values[percentile_min]:
           percentile_values[int(0.5 * percentile_min)] =  range_min
    else:
        percentile_values[0] = range_min

    print(f'Percentile_values {percentile_values.items()}')

    sorted_percentile_values = dict(sorted(percentile_values.items()))

    # Normalize percentile keys
    normalized_percentile_values = {}
    for key, value in sorted_percentile_values.items():
        percentile = float(key) / 100
        normalized_percentile_values[percentile] = value

    print(f'normalized_percentile_values: {normalized_percentile_values}')

    value_percentiles = {value: key for key, value in normalized_percentile_values.items()}

    print(f'value_percentiles: {value_percentiles}')


    range_min = scaling.get("range_min")
    range_max = scaling.get("range_max")
    zero_point = scaling.get("zero_point")


    # function for log scaled questions
    def generate_cdf_locations(range_min, range_max, zero_point):
        if zero_point is None:
            scale = lambda x: range_min + (range_max - range_min) * x
        else:
            deriv_ratio = (range_max - zero_point) / (range_min - zero_point)
            scale = lambda x: range_min + (range_max - range_min) * (
                deriv_ratio**x - 1
            ) / (deriv_ratio - 1)
        return [scale(x) for x in np.linspace(0, 1, 201)]

    cdf_xaxis = generate_cdf_locations(range_min, range_max, zero_point)

    print(f'range_min: {range_min}')
    print(f'range_max: {range_max}')
    print(f'zero_point: {zero_point}')
    print(f'cdf_axis: {cdf_xaxis}\n')


    def linear_interpolation(x_values, xy_pairs):
        # Sort the xy_pairs by x-values
        sorted_pairs = sorted(xy_pairs.items())

        # Extract sorted x and y values
        known_x = [pair[0] for pair in sorted_pairs]
        known_y = [pair[1] for pair in sorted_pairs]

        # Initialize the result list
        y_values = []

        for x in x_values:
            # Check if x is exactly in the known x values
            if x in known_x:
                y_values.append(known_y[known_x.index(x)])
            else:
                # Find the indices of the two nearest known x-values
                i = 0
                while i < len(known_x) and known_x[i] < x:
                    i += 1

                list_index_2 = i

                # If x is outside the range of known x-values, use the nearest endpoint
                if i == 0:
                    y_values.append(known_y[0])
                elif i == len(known_x):
                    y_values.append(known_y[-1])
                else:
                    # Perform linear interpolation
                    x0, x1 = known_x[i-1], known_x[i]
                    y0, y1 = known_y[i-1], known_y[i]

                    # Linear interpolation formula
                    y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
                    y_values.append(y)

        return y_values

    continuous_cdf = linear_interpolation(cdf_xaxis, value_percentiles)


    print(f'continuous_cdf: {continuous_cdf}')

    return continuous_cdf
def extract_option_probabilities_from_response(forecast_text: str, options) -> float:

    # Helper function that returns a list of tuples with numbers for all lines with Percentile
    def extract_option_probabilities(text):

        # Number extraction pattern
        number_pattern = r'-?\d+(?:,\d{3})*(?:\.\d+)?'

        results = []

        # Iterate through each line in the text
        for line in text.split('\n'):
            # Extract all numbers from the line
            numbers = re.findall(number_pattern, line)
            numbers_no_commas = [num.replace(',', '') for num in numbers]
            # Convert strings to float or int
            numbers = [float(num) if '.' in num else int(num) for num in numbers_no_commas]
            # Add the tuple of numbers to results
            if len(numbers) >= 1:
              last_number = numbers[-1]
              results.append(last_number)

        return results

    option_probabilities = extract_option_probabilities(forecast_text)

    NUM_OPTIONS = len(options)

    if len(option_probabilities) > 0:
        # return the last NUM_OPTIONS items
        return option_probabilities[-NUM_OPTIONS:]
    else:
        raise ValueError(
            f"Could not extract prediction from response: {forecast_text}"
        )


def generate_multiple_choice_forecast(options, option_probabilities) -> dict:
    """
    Returns: dict corresponding to the probabilities of each option.
    """
    print(f'options: {options}')
    print(f'option_probabilities: {option_probabilities}')

    # confirm that there is a probability for each option
    if len(options) != len(option_probabilities):
      raise ValueError(f"Number of options ({len(options)}) does not match number of probabilities ({len(option_probabilities)})")

    # Ensure we are using decimals
    total_sum = sum(option_probabilities)
    decimal_list = [x / total_sum for x in option_probabilities]

    def normalize_list(float_list):
        # Step 1: Clamp values
        clamped_list = [max(min(x, 0.99), 0.01) for x in float_list]

        # Step 2: Calculate the sum of all elements
        total_sum = sum(clamped_list)

        # Step 3: Normalize the list so that all elements add up to 1
        normalized_list = [x / total_sum for x in clamped_list]

        # Step 4: Adjust for any small floating-point errors
        adjustment = 1.0 - sum(normalized_list)
        normalized_list[-1] += adjustment

        return normalized_list

    normalized_option_probabilities = normalize_list(decimal_list)

    probability_yes_per_category = {}
    for i in range(len(options)):
        probability_yes_per_category[options[i]] = normalized_option_probabilities[i]

    print(f'probability_yes_per_category: {probability_yes_per_category}')

    return probability_yes_per_category

def list_questions(tournament_id, offset=0, count=50) -> list[dict]:
    """
    List (all details) {count} questions from the {tournament_id}
    """
    url_qparams = {
        "limit": count,
        "offset": offset,
        "has_group": "false",
        "order_by": "-activity",
        "forecast_type": ",".join([
            "binary",
            "multiple_choice",
            "numeric",
        ]),
        "project": tournament_id,
        "status": "open",
        "type": "forecast",
        "include_description": "true",
    }
    url = f"{API_BASE_URL2}/questions/"
    response = requests.get(url, **AUTH_HEADERS, params=url_qparams)
    if not response.ok:
        raise Exception(response.text)
    data = json.loads(response.content)
    return data

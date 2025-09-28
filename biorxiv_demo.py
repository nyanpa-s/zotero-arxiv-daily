import arxiv
import argparse
import os
import sys
import yaml
from dotenv import load_dotenv
load_dotenv(override=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pyzotero import zotero
from recommender import rerank_paper
from construct_email import render_email, send_email
from tqdm import trange,tqdm
from loguru import logger
from gitignore_parser import parse_gitignore
from tempfile import mkstemp
from paper import ArxivPaper, BiorxivPaper
from llm import set_global_llm
import feedparser
from datetime import datetime, timedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def get_zotero_corpus(id:str,key:str) -> list[dict]:
    zot = zotero.Zotero(id, 'user', key)
    collections = zot.everything(zot.collections())
    collections = {c['key']:c for c in collections}

    # Debug: Get all items first to see what we have
    all_items = zot.everything(zot.items())
    logger.debug(f"Total items in Zotero library: {len(all_items)}")

    # Debug: Check item types
    item_types = {}
    for item in all_items:
        item_type = item['data'].get('itemType', 'unknown')
        item_types[item_type] = item_types.get(item_type, 0) + 1
    logger.debug(f"Item types distribution: {item_types}")

    # Get filtered items
    corpus = zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
    logger.debug(f"Items after itemType filter: {len(corpus)}")

    # Debug: Check how many items have abstracts
    items_with_abstract = [c for c in corpus if c['data']['abstractNote'] != '']
    items_without_abstract = [c for c in corpus if c['data']['abstractNote'] == '']
    logger.debug(f"Items with abstract: {len(items_with_abstract)}")
    logger.debug(f"Items without abstract: {len(items_without_abstract)}")

    corpus = items_with_abstract
    def get_collection_path(col_key:str) -> str:
        if p := collections[col_key]['data']['parentCollection']:
            return get_collection_path(p) + '/' + collections[col_key]['data']['name']
        else:
            return collections[col_key]['data']['name']
    for c in corpus:
        paths = [get_collection_path(col) for col in c['data']['collections']]
        c['paths'] = paths
    return corpus

def filter_corpus(corpus:list[dict], pattern:str) -> list[dict]:
    _,filename = mkstemp()
    with open(filename,'w') as file:
        file.write(pattern)
    matcher = parse_gitignore(filename,base_dir='./')
    new_corpus = []
    for c in corpus:
        match_results = [matcher(p) for p in c['paths']]
        if not any(match_results):
            new_corpus.append(c)
    os.remove(filename)
    return new_corpus


def get_arxiv_paper(query:str, debug:bool=False) -> list[ArxivPaper]:
    client = arxiv.Client(num_retries=10,delay_seconds=10)
    feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
    if 'Feed error for query' in feed.feed.title:
        raise Exception(f"Invalid ARXIV_QUERY: {query}.")
    if not debug:
        papers = []
        all_paper_ids = [i.id.removeprefix("oai:arXiv.org:") for i in feed.entries if i.arxiv_announce_type == 'new']
        bar = tqdm(total=len(all_paper_ids),desc="Retrieving Arxiv papers")
        for i in range(0,len(all_paper_ids),50):
            search = arxiv.Search(id_list=all_paper_ids[i:i+50])
            batch = [ArxivPaper(p) for p in client.results(search)]
            bar.update(len(batch))
            papers.extend(batch)
        bar.close()

    else:
        logger.debug("Retrieve 5 arxiv papers regardless of the date.")
        search = arxiv.Search(query='cat:cs.AI', sort_by=arxiv.SortCriterion.SubmittedDate)
        papers = []
        for i in client.results(search):
            papers.append(ArxivPaper(i))
            if len(papers) == 5:
                break

    return papers

def get_biorxiv_paper(query: str, debug: bool = False) -> list[BiorxivPaper]:
    """
    Retrieve papers from bioRxiv API.
    
    Args:
        query: The search query/category
        debug: Whether to run in debug mode
        
    Returns:
        A list of BiorxivPaper objects
        
    Raises:
        requests.RequestException: If the API request fails
        ValueError: If the query is invalid
    """
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.1)
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    if not debug:
        today = datetime.now()
        yesterday = today - timedelta(days=1)
        formatted_date = today.strftime("%Y-%m-%d")
        formatted_yesterday = yesterday.strftime("%Y-%m-%d")
        if "+" in query:
            queries = query.split("+")
        else:
            queries = [query]

        papers = []
        for query in queries:
            url = f"https://api.biorxiv.org/details/biorxiv/{formatted_yesterday}/{formatted_date}?category={query}"
            logger.info(f"Retrieving biorxiv papers from {url}...")
            response = requests.get(url)
            if response.status_code != 200:
                raise Exception(f"Invalid URL format: {url}.")
            data = response.json()
            for i in data['collection']:
                if i['doi'] == '':
                    continue
                paper = BiorxivPaper(i)
                papers.append(paper)
    else:
        url = "https://api.biorxiv.org/details/biorxiv/2025-03-21/2025-03-28?category=cell_biology"
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Invalid BIORXIV_QUERY: {query}.")
        data = response.json()
        logger.debug("Retrieve 5 biorxiv papers regardless of the date.")
        papers = []
        for i in data['collection']:
            if i['doi'] == '':
                continue
            paper = BiorxivPaper(i)
            papers.append(paper)
            if len(papers) == 5:
                break
    return papers


parser = argparse.ArgumentParser(description='Recommender system for academic papers')

def add_argument(*args, **kwargs):
    def get_env(key:str,default=None):
        # handle environment variables generated at Workflow runtime
        # Unset environment variables are passed as '', we should treat them as None
        v = os.environ.get(key)
        if v == '' or v is None:
            return default
        return v
    parser.add_argument(*args, **kwargs)
    arg_full_name = kwargs.get('dest',args[-1][2:])
    env_name = arg_full_name.upper()
    env_value = get_env(env_name)
    if env_value is not None:
        #convert env_value to the specified type
        if kwargs.get('type') == bool:
            env_value = env_value.lower() in ['true','1']
        else:
            env_value = kwargs.get('type')(env_value)
        parser.set_defaults(**{arg_full_name:env_value})


if __name__ == '__main__':
    
    add_argument('--zotero_id', type=str, help='Zotero user ID')
    add_argument('--zotero_key', type=str, help='Zotero API key')
    add_argument('--zotero_ignore',type=str,help='Zotero collection to ignore, using gitignore-style pattern.')
    add_argument('--send_empty', type=bool, help='If get no arxiv paper, send empty email',default=False)
    add_argument('--max_paper_num', type=int, help='Maximum number of papers to recommend',default=50)
    add_argument('--max_biorxiv_num', type=int, help='Maximum number of biorxiv papers to recommend',default=50)
    add_argument('--arxiv_query', type=str, help='Arxiv search query')
    add_argument('--biorxiv_query', type=str, help='Biorxiv search category')
    add_argument('--smtp_server', type=str, help='SMTP server')
    add_argument('--smtp_port', type=int, help='SMTP port')
    add_argument('--sender', type=str, help='Sender email address')
    add_argument('--receiver', type=str, help='Receiver email address')
    add_argument('--sender_password', type=str, help='Sender email password')
    add_argument(
        "--use_llm_api",
        type=bool,
        help="Use OpenAI API to generate TLDR",
        default=False,
    )
    add_argument(
        "--openai_api_key",
        type=str,
        help="OpenAI API key",
        default=None,
    )
    add_argument(
        "--openai_api_base",
        type=str,
        help="OpenAI API base URL",
        default="https://api.openai.com/v1",
    )
    add_argument(
        "--model_name",
        type=str,
        help="LLM Model Name",
        default="gpt-4o",
    )
    add_argument(
        "--language",
        type=str,
        help="Language of TLDR",
        default="English",
    )
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()
    # load arguments from .yml file
    if os.path.exists("config.yml"):
        with open("config.yml", "r") as f:
            config = yaml.safe_load(f)
            # load arguments to args
            for key, value in config.items():
                if hasattr(args, key):
                    setattr(args, key, value)
            
    assert (
        not args.use_llm_api or args.openai_api_key is not None
    )  # If use_llm_api is True, openai_api_key must be provided
    if args.debug:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG")
        logger.debug("Debug mode is on.")
    else:
        logger.remove()
        logger.add(sys.stdout, level="INFO")

    logger.info("Retrieving Zotero corpus...")
    corpus = get_zotero_corpus(args.zotero_id, args.zotero_key)
    logger.info(f"Retrieved {len(corpus)} papers from Zotero (count: {str(len(corpus)).replace('0', 'zero').replace('1', 'one').replace('2', 'two').replace('3', 'three').replace('4', 'four').replace('5', 'five').replace('6', 'six').replace('7', 'seven').replace('8', 'eight').replace('9', 'nine')}).")

    # 显示获取到的文献详情
    logger.info("Retrieved papers details:")
    for i, paper in enumerate(corpus, 1):
        title = paper['data'].get('title', 'No Title')
        item_type = paper['data'].get('itemType', 'Unknown')
        date_added = paper['data'].get('dateAdded', 'Unknown')
        abstract_length = len(paper['data'].get('abstractNote', ''))
        logger.info(f"{i:2d}. [{item_type}] {title[:80]}{'...' if len(title) > 80 else ''}")
        logger.info(f"     Added: {date_added}, Abstract length: {abstract_length} chars")
    if args.zotero_ignore:
        # 显示完整的过滤规则，避免被 GitHub 屏蔽
        ignore_rules = args.zotero_ignore.replace('/', '_SLASH_').replace('*', '_STAR_')
        logger.info(f"Applying ignore rules: {ignore_rules}")
        logger.info(f"Original ignore pattern length: {len(args.zotero_ignore)} characters")

        original_count = len(corpus)
        corpus = filter_corpus(corpus, args.zotero_ignore)
        filtered_count = len(corpus)
        removed_count = original_count - filtered_count

        logger.info(f"Filtering results: original={original_count}, removed={removed_count}, remaining={filtered_count}")
        logger.info(f"Remaining {len(corpus)} papers after filtering (count: {str(len(corpus)).replace('0', 'zero').replace('1', 'one').replace('2', 'two').replace('3', 'three').replace('4', 'four').replace('5', 'five').replace('6', 'six').replace('7', 'seven').replace('8', 'eight').replace('9', 'nine')}).")
    logger.info("Retrieving Biorxiv papers...")
    papers = get_arxiv_paper(args.arxiv_query, args.debug)
    biorxiv_papers = get_biorxiv_paper(args.biorxiv_query, args.debug)
    logger.info(f"Retrieved {len(papers)} papers from Arxiv.")
    logger.info(f"Retrieved {len(biorxiv_papers)} papers from Biorxiv.")
    if len(papers) == 0 and len(biorxiv_papers) == 0:
        logger.info("No new papers found. Yesterday maybe a holiday and no one submit their work :). If this is not the case, please check the ARXIV_QUERY.")
        if not args.send_empty:
          exit(0)
    else:
        logger.info("Reranking papers...")
        papers, biorxiv_papers = rerank_paper(papers, biorxiv_papers, corpus)
        # biorxiv_papers = rerank_biorxiv_paper(papers, corpus)
        if args.max_paper_num != -1 and args.max_paper_num < len(papers):
            papers = papers[:args.max_paper_num]
        if args.max_biorxiv_num != -1 and args.max_biorxiv_num < len(biorxiv_papers):
            biorxiv_papers = biorxiv_papers[:args.max_biorxiv_num]
        if args.use_llm_api:
            logger.info("Using OpenAI API as global LLM.")
            set_global_llm(api_key=args.openai_api_key, base_url=args.openai_api_base, model=args.model_name, lang=args.language)
        else:
            logger.info("Using Local LLM as global LLM.")
            set_global_llm(lang=args.language)

    html = render_email(papers, biorxiv_papers)
    logger.info("Sending email...")
    send_email(args.sender, args.receiver, args.sender_password, args.smtp_server, args.smtp_port, html)
    logger.success("Email sent successfully! If you don't receive the email, please check the configuration and the junk box.")


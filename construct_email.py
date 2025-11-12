from paper import ArxivPaper, BiorxivPaper
import math
from tqdm import tqdm
from email.header import Header
from email.mime.text import MIMEText
from email.utils import parseaddr, formataddr
import smtplib
import datetime
import time
from loguru import logger

framework = """
<!DOCTYPE HTML>
<html>
<head>
  <style>
    .star-wrapper {
      font-size: 1.3em; /* 调整星星大小 */
      line-height: 1; /* 确保垂直对齐 */
      display: inline-flex;
      align-items: center; /* 保持对齐 */
    }
    .half-star {
      display: inline-block;
      width: 0.5em; /* 半颗星的宽度 */
      overflow: hidden;
      white-space: nowrap;
      vertical-align: middle;
    }
    .full-star {
      vertical-align: middle;
    }
    .stats-section {
      background-color: #f5f5f5;
      border: 1px solid #ddd;
      border-radius: 8px;
      padding: 16px;
      margin-top: 20px;
      font-family: Arial, sans-serif;
    }
    .stats-title {
      font-size: 18px;
      font-weight: bold;
      color: #333;
      margin-bottom: 10px;
    }
    .stats-content {
      font-size: 14px;
      color: #666;
      line-height: 1.6;
    }
  </style>
</head>
<body>

<h1>Arxiv Papers</h1>
<div>
    __CONTENT-ARXIV__
</div>

<h1>BioRxiv Papers</h1>
<div>
    __CONTENT-BIORXIV__
</div>

__STATS-SECTION__

<br><br>
<div>
To unsubscribe, remove your email in your Github Action setting.
</div>

</body>
</html>
"""

def get_empty_html():
  block_template = """
  <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
  <tr>
    <td style="font-size: 20px; font-weight: bold; color: #333;">
        No Papers Today. Take a Rest!
    </td>
  </tr>
  </table>
  """
  return block_template

def get_block_html(title:str, authors:str, rate:str,arxiv_id:str, abstract:str, pdf_url:str, code_url:str=None, affiliations:str=None):
    code = f'<a href="{code_url}" style="display: inline-block; text-decoration: none; font-size: 14px; font-weight: bold; color: #fff; background-color: #5bc0de; padding: 8px 16px; border-radius: 4px; margin-left: 8px;">Code</a>' if code_url else ''
    block_template = """
    <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
    <tr>
        <td style="font-size: 20px; font-weight: bold; color: #333;">
            {title}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #666; padding: 8px 0;">
            {authors}
            <br>
            <i>{affiliations}</i>
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>Relevance:</strong> {rate}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>arXiv ID:</strong> <a href="https://arxiv.org/abs/{arxiv_id}" target="_blank">{arxiv_id}</a>
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>TLDR:</strong> {abstract}
        </td>
    </tr>

    <tr>
        <td style="padding: 8px 0;">
            <a href="{pdf_url}" style="display: inline-block; text-decoration: none; font-size: 14px; font-weight: bold; color: #fff; background-color: #d9534f; padding: 8px 16px; border-radius: 4px;">PDF</a>
            {code}
        </td>
    </tr>
</table>
"""
    return block_template.format(title=title, authors=authors,rate=rate,arxiv_id=arxiv_id, abstract=abstract, pdf_url=pdf_url, code=code, affiliations=affiliations)

def get_stats_html(zotero_stats:dict, arxiv_count:int, biorxiv_count:int):
    if not zotero_stats:
        return ""

    # Format item types distribution
    item_types_str = ", ".join([f"{k}: {v}" for k, v in zotero_stats['item_types'].items()])

    # Format items without abstract
    without_abstract_list = ""
    if zotero_stats['without_abstract_items']:
        for i, item in enumerate(zotero_stats['without_abstract_items'], 1):
            title = item['data'].get('title', 'No Title')
            item_type = item['data'].get('itemType', 'Unknown')
            without_abstract_list += f"{i}. [{item_type}] {title}<br>"

    stats_html = f"""
    <div class="stats-section">
        <div class="stats-title">Library Statistics</div>
        <div class="stats-content">
            <strong>Total Zotero library:</strong> {item_types_str}<br>
            <strong>After itemType filter:</strong> {zotero_stats['after_filter']}, with abstract: {zotero_stats['with_abstract']}, without abstract: {zotero_stats['without_abstract']}<br>
            {f"<strong>Items without abstract:</strong><br>{without_abstract_list}" if without_abstract_list else ""}
            <strong>Retrieved {arxiv_count} papers from Arxiv, {biorxiv_count} papers from Biorxiv</strong>
        </div>
    </div>
    """
    return stats_html

def get_stars(score:float):
    full_star = '<span class="full-star">⭐</span>'
    half_star = '<span class="half-star">⭐</span>'
    low = 6
    high = 8
    if score <= low:
        return ''
    elif score >= high:
        return full_star * 5
    else:
        interval = (high-low) / 10
        star_num = math.ceil((score-low) / interval)
        full_star_num = int(star_num/2)
        half_star_num = star_num - full_star_num * 2
        return '<div class="star-wrapper">'+full_star * full_star_num + half_star * half_star_num + '</div>'


def render_email(papers:list[ArxivPaper], papers_biorxiv:list[BiorxivPaper], zotero_stats:dict=None, arxiv_count:int=0, biorxiv_count:int=0):
    parts = []
    if len(papers) == 0 :
        framework1 = framework.replace('__CONTENT-ARXIV__', get_empty_html())
    else:
        for p in tqdm(papers,desc='Rendering Email'):
            rate = get_stars(p.score)
            authors = [a.name for a in p.authors[:5]]
            authors = ', '.join(authors)
            if len(p.authors) > 5:
                authors += ', ...'
            if p.affiliations is not None:
                affiliations = p.affiliations[:5]
                affiliations = ', '.join(affiliations)
                if len(p.affiliations) > 5:
                    affiliations += ', ...'
            else:
                affiliations = 'Unknown Affiliation'
            parts.append(get_block_html(p.title, authors,rate,p.arxiv_id,p.tldr, p.pdf_url, p.code_url, affiliations))

        content = '<br>' + '</br><br>'.join(parts) + '</br>'
        framework1 = framework.replace('__CONTENT-ARXIV__', content)


    parts = []
    if len(papers_biorxiv) == 0:
        framework2 = framework1.replace('__CONTENT-BIORXIV__', get_empty_html())
    else:
        for p in tqdm(papers_biorxiv,desc='Rendering Email'):
            rate = get_stars(p.score)
            authors = [a for a in p.authors[:5]]
            authors = ', '.join(authors)
            if len(p.authors) > 5:
                authors += ', ...'
            if p.institution is not None:
                affiliations = p.institution
            else:
                affiliations = 'Unknown Affiliation'
            parts.append(get_block_html(p.title, authors,rate,p.biorxiv_id,p.tldr, p.paper_url, p.code_url, affiliations))

        content = '<br>' + '</br><br>'.join(parts) + '</br>'
        framework2 = framework1.replace('__CONTENT-BIORXIV__', content)

    # Add statistics section
    stats_html = get_stats_html(zotero_stats, arxiv_count, biorxiv_count)
    return framework2.replace('__STATS-SECTION__', stats_html)


def send_email(sender:str, receiver:str, password:str,smtp_server:str,smtp_port:int, html:str,):
    def _format_addr(s):
        name, addr = parseaddr(s)
        return formataddr((Header(name, 'utf-8').encode(), addr))

    msg = MIMEText(html, 'html', 'utf-8')
    msg['From'] = _format_addr('Github Action <%s>' % sender)
    msg['To'] = _format_addr('You <%s>' % receiver)
    today = datetime.datetime.now().strftime('%Y/%m/%d')
    msg['Subject'] = Header(f'Daily arXiv {today}', 'utf-8').encode()

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
    except Exception as e:
        logger.warning(f"Failed to use TLS. {e}")
        logger.warning(f"Try to use SSL.")
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)

    server.login(sender, password)
    server.sendmail(sender, [receiver], msg.as_string())
    server.quit()

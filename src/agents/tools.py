"""Financial tools for the IBEX35 agent."""

from datetime import datetime, timedelta

import yfinance as yf
from langchain_core.tools import StructuredTool

from src.ingestion.pdf_loader import COMPANY_TICKER_MAP
from src.logging_config import get_logger

logger = get_logger(__name__)


def get_stock_price(company_name: str) -> str:
    """
    Get the current stock price and basic info for an IBEX35 company.

    Args:
        company_name: Company name (e.g. 'IBERDROLA', 'SANTANDER', 'INDITEX')

    Returns:
        String with current price, change, volume and 52-week range.
    """
    ticker_symbol = COMPANY_TICKER_MAP.get(company_name.upper())
    if not ticker_symbol:
        available = ", ".join(sorted(COMPANY_TICKER_MAP.keys()))
        return f"Company '{company_name}' not found. Available: {available}"

    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.fast_info
        hist = ticker.history(period="2d")

        if hist.empty:
            return f"No price data available for {company_name} ({ticker_symbol})"

        current_price = hist["Close"].iloc[-1]
        prev_close = hist["Close"].iloc[-2] if len(hist) > 1 else current_price
        change_pct = ((current_price - prev_close) / prev_close) * 100

        return (
            f"{company_name} ({ticker_symbol})\n"
            f"Price: €{current_price:.2f}\n"
            f"Change: {change_pct:+.2f}%\n"
            f"52w High: €{info.year_high:.2f}\n"
            f"52w Low: €{info.year_low:.2f}\n"
            f"Market Cap: €{info.market_cap / 1e9:.1f}B"
        )
    except Exception as exc:
        logger.warning("stock_price_error", company=company_name, error=str(exc))
        return f"Could not retrieve price for {company_name}: {exc}"


def get_price_history(company_name: str, days: int = 30) -> str:
    """
    Get historical price data for an IBEX35 company.

    Args:
        company_name: Company name (e.g. 'BBVA', 'TELEFONICA')
        days: Number of days of history (default 30, max 365)

    Returns:
        Summary of price evolution over the period.
    """
    ticker_symbol = COMPANY_TICKER_MAP.get(company_name.upper())
    if not ticker_symbol:
        return f"Company '{company_name}' not found in IBEX35."

    days = min(max(days, 1), 365)

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(start=start_date.strftime("%Y-%m-%d"))

        if hist.empty:
            return f"No historical data for {company_name}"

        start_price = hist["Close"].iloc[0]
        end_price = hist["Close"].iloc[-1]
        max_price = hist["Close"].max()
        min_price = hist["Close"].min()
        total_return = ((end_price - start_price) / start_price) * 100

        return (
            f"{company_name} ({ticker_symbol}) — Last {days} days\n"
            f"Start: €{start_price:.2f} → Current: €{end_price:.2f}\n"
            f"Return: {total_return:+.2f}%\n"
            f"High: €{max_price:.2f} | Low: €{min_price:.2f}"
        )
    except Exception as exc:
        logger.warning("price_history_error", company=company_name, error=str(exc))
        return f"Could not retrieve history for {company_name}: {exc}"


def list_ibex35_companies() -> str:
    """
    List all IBEX35 companies available in the system with their tickers.

    Returns:
        Formatted list of company names and ticker symbols.
    """
    lines = ["IBEX35 Companies in the system:\n"]
    for company, ticker in sorted(COMPANY_TICKER_MAP.items()):
        lines.append(f"  • {company:<20} {ticker}")
    return "\n".join(lines)


def compare_companies_price(companies: list[str], days: int = 30) -> str:
    """
    Compare stock price performance of multiple IBEX35 companies.

    Args:
        companies: List of company names to compare (e.g. ['IBERDROLA', 'ENDESA', 'NATURGY'])
        days: Period in days for comparison (default 30)

    Returns:
        Comparison table with returns for each company.
    """
    results = []
    for company in companies[:5]:  # Cap at 5 to avoid rate limits
        ticker_symbol = COMPANY_TICKER_MAP.get(company.upper())
        if not ticker_symbol:
            continue
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            hist = yf.Ticker(ticker_symbol).history(start=start_date.strftime("%Y-%m-%d"))
            if not hist.empty:
                ret = ((hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0]) * 100
                results.append((company, ticker_symbol, ret, hist["Close"].iloc[-1]))
        except Exception:
            continue

    if not results:
        return "Could not retrieve data for the requested companies."

    results.sort(key=lambda x: x[2], reverse=True)
    header = f"{'Company':<20} {'Ticker':<10} {'Price':>10} {f'Return ({days}d)':>12}"
    separator = "-" * 55
    rows = [
        f"{c:<20} {t:<10} €{p:>9.2f} {r:>+11.2f}%"
        for c, t, r, p in results
    ]
    return "\n".join([header, separator, *rows])


# LangChain StructuredTools
stock_price_tool = StructuredTool.from_function(
    func=get_stock_price,
    name="get_stock_price",
    description="Get current stock price and basic market data for an IBEX35 company.",
)

price_history_tool = StructuredTool.from_function(
    func=get_price_history,
    name="get_price_history",
    description="Get historical stock price evolution for an IBEX35 company over N days.",
)

list_companies_tool = StructuredTool.from_function(
    func=list_ibex35_companies,
    name="list_ibex35_companies",
    description="List all IBEX35 companies available in the system with their ticker symbols.",
)

compare_price_tool = StructuredTool.from_function(
    func=compare_companies_price,
    name="compare_companies_price",
    description="Compare stock price performance of multiple IBEX35 companies over a given period.",
)

ALL_TOOLS = [stock_price_tool, price_history_tool, list_companies_tool, compare_price_tool]

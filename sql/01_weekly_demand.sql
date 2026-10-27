-- Weekly demand per SKU: positive, non-cancellation lines only.
-- One row per (stock_code, week_start), week_start = Monday of the ISO week.
CREATE OR REPLACE TABLE weekly_demand AS
WITH coverage AS (
    SELECT CAST(MIN(invoice_date) AS DATE) AS first_day,
           CAST(MAX(invoice_date) AS DATE) AS last_day FROM transactions
)
SELECT
    stock_code,
    CAST(date_trunc('week', invoice_date) AS DATE) AS week_start,
    SUM(quantity)              AS units,
    COUNT(DISTINCT invoice_no) AS orders,
    SUM(line_amount)           AS revenue
FROM transactions
  CROSS JOIN coverage
WHERE quantity > 0
  AND unit_price > 0
  AND NOT is_cancellation
  -- Do not treat the source's partial first/last weeks as complete demand weeks.
  AND date_trunc('week', invoice_date) >= first_day
  AND date_trunc('week', invoice_date) + INTERVAL 6 DAY <= last_day
GROUP BY stock_code, week_start;

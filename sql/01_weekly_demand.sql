-- Weekly demand per SKU: positive, non-cancellation lines only.
-- One row per (stock_code, week_start), week_start = Monday of the ISO week.
CREATE OR REPLACE TABLE weekly_demand AS
SELECT
    stock_code,
    CAST(date_trunc('week', invoice_date) AS DATE) AS week_start,
    SUM(quantity)              AS units,
    COUNT(DISTINCT invoice_no) AS orders,
    SUM(line_amount)           AS revenue
FROM transactions
WHERE quantity > 0
  AND NOT is_cancellation
GROUP BY stock_code, week_start;

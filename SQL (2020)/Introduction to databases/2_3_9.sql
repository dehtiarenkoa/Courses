#Выведите список всех источников клиентов и суммарный объем заказов по каждому источнику. Результат должен включать также записи для источников, по которым не было заказов.
SELECT 
    source.name AS source_name, sum(sale.sale_sum) AS sale_sum
FROM
    sale
        JOIN
    client ON sale.client_id = client.id
        RIGHT JOIN
    source ON source.id = client.source_id
GROUP BY source.name
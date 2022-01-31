#Выведите список источников, из которых не было клиентов, 
#либо клиенты пришедшие из которых не совершали заказов или отказывались от заказов. 
#Под клиентами, которые отказывались от заказов, необходимо понимать клиентов, 
#у которых есть заказы, которые на момент выполнения запроса находятся в состоянии 'rejected'. 
#В запросе необходимо использовать оператор UNION для объединения выборок по разным условиям.
SELECT 
    source.name AS source_name
FROM
    source
        JOIN
    client
WHERE
    source.id NOT IN (SELECT 
            client.source_id
        FROM
            client) 
UNION SELECT 
    source.name AS source_name
FROM
    source
        JOIN
    client ON client.source_id = source.id
        JOIN
    sale
WHERE
    client.id NOT IN (SELECT 
            sale.client_id
        FROM
            sale) 
UNION SELECT 
    source.name AS source_name
FROM
    source
        JOIN
    client ON client.source_id = source.id
        JOIN
    sale ON sale.client_id = client.id
        JOIN
    status ON sale.status_id = status.id
WHERE
    status.name = 'rejected'
GROUP BY source.name

/*
doesnt work properly
SELECT source.name AS source_name FROM source
  WHERE NOT EXISTS (SELECT * FROM source, client WHERE source.id = client.source_id)
  union
  SELECT source.name AS source_name FROM source
   JOIN
    client ON client.source_id = source.id
  WHERE NOT EXISTS (SELECT * FROM client, sale WHERE client.id = sale.client_id)
UNION SELECT 
    source.name AS source_name
FROM
    source
        JOIN
    client ON client.source_id = source.id
        JOIN
    sale ON sale.client_id = client.id
        JOIN
    status ON sale.status_id = status.id
WHERE
    status.name = 'rejected'
GROUP BY source.name
*/
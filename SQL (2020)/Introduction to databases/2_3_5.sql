#Выведите список клиентов (имя, фамилия) и количество заказов данных клиентов, имеющих статус "new".

select 
client.first_name as first_name,
client.last_name as last_name,
count(status.name) as new_sale_num
from status
inner join sale on sale.status_id=status.id
inner join client on sale.client_id=client.id
where status.name="new"
group by last_name, first_name
#order by good.name, category.name
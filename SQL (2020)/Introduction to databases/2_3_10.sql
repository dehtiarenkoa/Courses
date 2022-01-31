#Выведите названия товаров, которые относятся к категории 'Cakes' или фигурируют в заказах текущий статус которых 'delivering'. Результат не должен содержать одинаковых записей. В запросе необходимо использовать оператор UNION для объединения выборок по разным условиям.
select good.name as good_name
from good
join category_has_good on category_has_good.good_id=good.id
join category on category_has_good.category_id=category.id
where category.name="Cakes"
union
select good.name as good_name
from good 
join sale_has_good on sale_has_good.good_id=good.id
join sale on sale_has_good.sale_id=sale.id
join status on status.id=sale.status_id
where status.name="delivering"
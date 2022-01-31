#Выведите список всех категорий продуктов и количество продаж товаров, относящихся к данной категории. Под количеством продаж товаров подразумевается суммарное количество единиц товара данной категории, фигурирующих в заказах с любым статусом.
select category.name as category_name, count(sale_has_good.sale_id) as sale_num
from category
left join category_has_good on category_has_good.category_id=category.id
left join good on good.id=category_has_good.good_id
left join sale_has_good on sale_has_good.good_id=good.id
#join sale on sale_has_good.sale_id=sale.id
group by category_name
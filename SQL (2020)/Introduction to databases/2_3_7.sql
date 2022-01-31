#Выведите список товаров с названиями товаров и названиями категорий, в том числе товаров, не принадлежащих ни одной из категорий.
select good.name as good_name, category.name as category_name
from good 
left join category_has_good on category_has_good.good_id=good.id
left join category on category_has_good.category_id=category.id
group by good.name, category.name

#Выведите список товаров с названиями категорий, в том числе товаров, не принадлежащих ни к одной из категорий, в том числе категорий не содержащих ни одного товара.
select good.name as good_name, category.name as category_name
from good 
left join category_has_good on category_has_good.good_id=good.id
left join category on category_has_good.category_id=category.id
union
select good.name as good_name, category.name as category_name
from good 
right join category_has_good on category_has_good.good_id=good.id
right join category on category_has_good.category_id=category.id

group by good.name, category.name
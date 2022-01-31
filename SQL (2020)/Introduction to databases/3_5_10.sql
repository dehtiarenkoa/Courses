/*База данных магазина `store` следующей структуры...
В таблице 'client' ограничение внешнего ключа называется 'fk_client_source1', определенное на поле 'source_id'.

Задание
Добавьте в таблицу 'sale_has_good' следующие поля:

    Название: `num`, тип данных: INT, 
    возможность использования неопределенного значения: Нет
    Название: `price`, тип данных: DECIMAL(18,2), 
    возможность использования неопределенного значения: Нет

NB! При выполнении ALTER TABLE не следует указывать название схемы.
*/
alter table sale_has_good
add column num INT not null,
add column price DECIMAL(18,2) not null
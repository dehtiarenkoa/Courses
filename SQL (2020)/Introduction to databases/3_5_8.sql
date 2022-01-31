/*База данных магазина `store` следующей структуры...
В таблице 'client' ограничение внешнего ключа называется 'fk_client_source1', определенное на поле 'source_id'.

Задание
Удалите из таблицы 'client' поля 'code' и 'source_id'.

NB! Для удаления поля, являющегося внешним ключом, необходимо удалить 
ограничение внешнего ключа оператором 'DROP FOREIGN KEY <fk_name>', 
для данного задание имя первичного ключа: fk_client_source1. 
Удаление ограничения внешнего ключа и поля таблицы необходимо производить 
в рамках одного вызова ALTER TABLE.

NB! При выполнении ALTER TABLE не следует указывать название схемы.
*/
alter table client
drop column code,
drop column source_id,
drop foreign key fk_client_source1

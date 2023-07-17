## Design a RDMS database using OOP’s principles that supports the below operations:

1. It should be possible to create/truncate/delete tables in a database.
2. It should be possible to filter and display records whose column values match a given value.
3. The supported column types are string and int and should be extendable for other column types.
4. It should be possible to insert records in a table.
5. It should be possible to print all records in a table.


Now lets discuss these classes & their attributes:

1. `Database` : This class should define the database name & must store the tables present in this database.
```java
private String databaseName;
private Map<String, Table> tableMap = new HashMap ();
```
Also, this will be will serve as client interface for all the Database operations

2. `Column`: This class will store the column name & its type. We will define the column type as enum values
```java
private String columnName;
public enum Type {INT,STRING};
private Type columnType;
```

3. `Row` : This class will store the rowId & column data
```java
private Integer rowId;
private Map<Column, Object> columnData;
```

4. `Table`: This class should define the table name, must store the column name & its type, records(rows) & it must also store auto increment Id. It must ensure that this id is incremented by 1 on each row insertion & concurrent requests are handled as well.
```java
private Integer autoIncrementId;
private String name;
private Map<String,Column> columnMap = new HashMap ();
List<Row> rows = new ArrayList ();
```
We have set up our classes now let’s jump over the functional requirements.

Let’s start with the easy part, lets implement Row and Column classes as they are straight forward


### Column:
```java
package Database;

public class Column {

    private String columnName;
    public enum Type {INT,STRING};
    private Type columnType;

    public Column(String columnName, Type columnType) {

        this.columnName = columnName;
        this.columnType = columnType;
    }

    public String getColumnName() {
        return this.columnName;
    }
}

```

### Row:
```java
package Database;

import java.util.Map;

public class Row {

    private Integer rowId;
    private Map<Column, Object> columnData;

    public Row(Integer rowId, Map<Column, Object> columnData) {

        this.rowId = rowId;
        this.columnData = columnData;
    }

    public Integer getRowId() {

        return this.rowId;
    }

    public Map<Column, Object> getColumnData() {

        return this.columnData;
    }
}
```

Let's create methods for Table:

```java
package Database;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Table {

    private Integer autoIncrementId;
    private String name;
    private Map<String,Column> columnMap = new HashMap ();
    List<Row> rows = new ArrayList ();

    public Table(String tableName, List<Column> columns) {

        this.autoIncrementId = 1;
        this.name = tableName;
        populateColumnMap(columns);
    }

    protected void truncateRows() {

        this.rows.clear();
    }

    protected void insertRow(Map<Column, Object> columnValues) {

        for(Column column:columnValues.keySet()) {
            if(!checkIfColumnExists(column.getColumnName())) return;
        }
        Integer rowId = getAutoIncrementId();
        Map<Column, Object> columnData = new HashMap (columnValues);
        Row row = new Row(rowId, columnData);
        this.rows.add(row);
    }

    protected void printRows() {

        System.out.println("Printing all rows for Table: "+this.name);
        printRecords(this.rows);
    }

    protected void getRecordsByColumnValue(Column column, Object value) {

        List<Row> rows = new ArrayList ();
        for(Row row:this.rows) {
            Object columnValue = row.getColumnData().get(column);
            if(columnValue.equals(value)) {
                rows.add(row);
            }
        }
        System.out.println("Printing matching rows for Table: "+this.name);
        printRecords(rows);
    }

    private void printRecords(List<Row> rows) {

        System.out.print("\t");
        for(Map.Entry<String,Column> entry : this.columnMap.entrySet()) {
            System.out.print("\t"+entry.getKey()+"\t");
        }
        for(Row row: rows) {
            System.out.print("\n\t"+row.getRowId()+".");
            for(Map.Entry<Column, Object> entry : row.getColumnData().entrySet()) {
                System.out.print("\t"+entry.getValue()+"\t");
            }
        }
        System.out.print("\n");
    }

    private void populateColumnMap(List<Column> columns) {

        for(Column column: columns) {
            columnMap.put(column.getColumnName(),column);
        }
    }

    private synchronized Integer getAutoIncrementId() {
        return this.autoIncrementId++;
    }

    private Boolean checkIfColumnExists(String columnName) {

        if(!columnMap.containsKey(columnName)) {
            System.out.println("TableName: "+this.name+" does not contains column: "+columnName);
            return false;
        }
        return true;
    }
}
```

## Database:
```java
package Database;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Database {

    private String databaseName;
    private Map<String, Table> tableMap = new HashMap ();

    public Database(String databaseName) {
        this.databaseName = databaseName;
    }

    public void createTable(String tableName, List<Column> columns) {

        if(checkIfTableExists(tableName)) System.out.println("TableName: "+ tableName+" already exists!");
        Table table = new Table(tableName, columns);
        tableMap.put(tableName, table);
        return;
    }

    public void dropTable(String tableName) {

        if(!checkIfTableExists(tableName)) return;
        tableMap.remove(tableName);
        System.out.println("TableName: "+tableName+" dropped!");
        return;
    }

    public void truncate(String tableName) {

        if(!checkIfTableExists(tableName)) return;
        Table table = tableMap.get(tableName);
        table.truncateRows();
    }

    public void insertTableRows(String tableName, Map<Column, Object> columnValues) {

        if(!checkIfTableExists(tableName)) return;
        Table table = tableMap.get(tableName);
        table.insertRow(columnValues);
    }

    public void printTableAllRows(String tableName) {

        if(!checkIfTableExists(tableName)) return;
        Table table = tableMap.get(tableName);
        table.printRows();
    }

    public void filterTableRecordsByColumnValue(String tableName, Column column, Object value) {

        if(!checkIfTableExists(tableName)) return;
        Table table = tableMap.get(tableName);
        table.getRecordsByColumnValue(column, value);
    }

    private Boolean checkIfTableExists(String tableName) {

        if(!tableMap.containsKey(tableName)) {
            System.out.println("TableName: "+tableName+" does not exists");
            return false;
        }
        return true;
    }
}
```

### Testing:
```java
import Database.Database;
import Database.Column;
import Database.Table;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Application {

    private static final String tableName = "Employee";

    public static void main(String[] args) {

        Column name = new Column("name", Column.Type.STRING);
        Column age = new Column("age", Column.Type.INT);
        Column salary = new Column("salary", Column.Type.INT);
        Database db = new Database("MyDB");
        List<Column> columns = new ArrayList();
        columns.add(name);
        columns.add(age);
        columns.add(salary);
        db.createTable(tableName,columns);
        Map<Column,Object> columnValues = new HashMap ();
        columnValues.put(name, "John");
        columnValues.put(age, 25);
        columnValues.put(salary, 10000);

        db.insertTableRows(tableName,columnValues);
        columnValues.clear();
        columnValues.put(name, "Kim");
        columnValues.put(age, 28);
        columnValues.put(salary, 12000);
        db.insertTableRows(tableName,columnValues);
        db.printTableAllRows(tableName);
        db.filterTableRecordsByColumnValue(tableName, age, 28);

        db.filterTableRecordsByColumnValue(tableName, name, "John");
        db.truncate(tableName);
        db.dropTable(tableName);
        db.printTableAllRows(tableName);
    }
}
```

- https://kishannigam.medium.com/designing-an-in-memory-rdms-database-91027751f95a

## Map<key, val>
- `.containsKey(key)`
- `.containsValue(val)`
- `.put(key, val)`
- `.remove(key)`
- `.clear()` : clear all elements

## Collections
- List, Set, Queue
- `contains(ele)`
- `.add(ele)`
- `.clear()` clear all elements

- Iterate over a key set of a map:
```java
Map<String, Object> myMap = new HashMap();

for (String key: myMap.keySet()) {
    if myList.contains(key) ...
```

## Convert between List<myObj> and ArrayList
- ArrayList is class that impl List. So any instance of ArrayList is an instance of List. You can directly assign an ArrayList to a List:
```java
List<myObj> l1 = new ArrayList<>();
ArrayList<myObj> l2 = new ArrayList<>(l1);
```

## Convert from array (fixed size) to ArrayList or List
**myObj[] --> List<myObj>**

```java
ArrayList<myObj> l1 = new ArrayList<>(Arrays.asList(myArr));
```

**myObj[] <-- List<myObj>**
```java
myObj[] l1 = list.toArray(new myObj[0])
```

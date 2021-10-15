# Missing value.
# Mixed date formats.
# Different representations of the same values.
# Formulas (e.g. summation).
# Duplicate values.
# Mixed numerical scales.
# Redundant data.
# Spelling errors.

# HOTEL BOOKING DATASET

using CSV
using DataFrames
hotel_data = CSV.read("/Users/Joel/Desktop/BIO397/week01.2/hotel_bookings.csv",DataFrame)
descript = describe(hotel_data)
print(descript)

# missing values?
sum(ismissing(hotel_data))
# There are no missing values in this dataset...

# Mixed data formats?
names(hotel_data)
hotel_data[!,:arrival_date_year]
hotel_data[!,:arrival_date_month]
hotel_data[!,:arrival_date_week_number]
hotel_data[!,:arrival_date_day_of_month]
hotel_data[!,:reservation_status_date]
# we can see that the reservation_status_date is in another format
# Let's fix this:
using Dates
month = Dict("January" => 1,"February" => 2, "March" => 3, "April" => 4, "May" => 5, "June" => 6,
"July" => 7, "August" => 8,"September"=>9,"October"=>10,"November"=>11,"December"=>12)

arrival_date = Date.(Dates.Year.(hotel_data[!,:arrival_date_year]),
Dates.Month.([month[i] for i in hotel_data[!,:arrival_date_month]]),Dates.Day.(hotel_data[!,:arrival_date_day_of_month]))
hotel_data.arrival_date  = arrival_date
hotel_data[!,:arrival_date]

# Different representations of the same values.
# There are no different representations of the same values.

# Formulas (e.g. summation).
colnames = names(hotel_data)
print(colnames)
# There are no variables that are means or other statistics of something.

# Duplicate values.
nonuniques = hotel_data[nonunique(hotel_data),:]
sort(nonuniques,:hotel)
ind = findall(nonunique(hotel_data))
hotel_data = delete!(hotel_data,ind)

# Mixed numerical scales.
print(describe(hotel_data))
sdd = select(hotel_data,[:adults,:is_canceled,:stays_in_week_nights])

sdf(x, m, s) = (x-m)/s

sdadults = sdf.(sdd.adults, mean(sdd.adults), std(sdd.adults))
sdis_canceled = sdf.(sdd.is_canceled, mean(sdd.is_canceled), std(sdd.is_canceled))
sdstays_in_week_nights = sdf.(sdd.stays_in_week_nights, mean(sdd.stays_in_week_nights), std(sdd.stays_in_week_nights))


# Redundant data.
sum(nonunique(hotel_data))
# we now have multiple columns for the arrival data --> i will delte them when i create
# a new clean dataset.


# Spelling errors.
names(hotel_data)
# here are no spelling errors...
unique(hotel_data.market_segment)
# no spelling errors
unique(hotel_data.country)
# no spelling errors


# CREATE AND SAVE A NEW CLEAN DATA DataFrame

clean_df = DataFrame(is_canceled=sdis_canceled,adults=sdadults,stays_in_week_nights=sdstays_in_week_nights,
arrival_date=hotel_data.arrival_date,country=hotel_data.country)
CSV.write("clean_data.csv", clean_df)
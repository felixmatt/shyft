from shyft import api
import unittest
import datetime as dt


class CalendarTestCase(unittest.TestCase):
    """Verify and illustrate the Calendar & utctime from the api core, using
    pyunit. Note that the Calendar not yet support local/dst semantics (but
    plan to do so) Nevertheless, keeping it here allow users of api-Core to
    use start practicing utctime/calendar perimeter.
    """
    def setUp(self):
        self.utc = api.Calendar()  # A utc calendar
        self.std = api.Calendar(3600)  # UTC+01

    def tearDown(self):
        self.utc = None
        self.std = None

    def test_trim_day(self):
        t = api.utctime_now()
        td = self.std.trim(t, api.Calendar.DAY)
        c = self.std.calendar_units(td)
        a = self.std.calendar_units(t)
        self.assertEqual(c.second, 0, 'incorrect seconds should be 0')
        self.assertEqual(c.minute, 0, 'trim day should set minutes to 0')
        self.assertEqual(c.hour, 0, 'trim day should set hours to 0')
        self.assertEqual(a.year, c.year, 'trim day Should leave same year')
        self.assertEqual(a.month, c.month, 'trim day  Should leave month')
        self.assertEqual(a.day, c.day, 'should leave same day')

    def test_conversion_roundtrip(self):
        c1 = api.YMDhms(2000, 01, 02, 03, 04, 05)
        t1 = self.std.time(c1)
        c2 = self.std.calendar_units(t1)
        self.assertEqual(c1.year, c2.year, 'roundtrip should keep year')
        self.assertEqual(c1.month, c2.month)
        self.assertEqual(c1.day, c2.day)
        self.assertEqual(c1.hour, c2.hour)
        self.assertEqual(c1.second, c2.second)

    def test_utctime_now(self):
        a = api.utctime_now()
        x = dt.datetime.utcnow()
        b = self.utc.time(api.YMDhms(x.year, x.month, x.day,
                          x.hour, x.minute, x.second))
        self.assertLess(abs(a - b), 2, 'Should be less than 2 seconds')

    def test_utc_time_to_string(self):
        c1 = api.YMDhms(2000, 01, 02, 03, 04, 05)
        t = self.std.time(c1)
        s = self.std.to_string(t)
        self.assertEqual(s, "2000.01.02T03:04:05")

    def test_UtcPeriod_to_string(self):
        c1 = api.YMDhms(2000, 01, 02, 03, 04, 05)
        t = self.utc.time(c1)
        p = api.UtcPeriod(t, t + api.deltahours(1))
        s = p.to_string()
        self.assertEqual(s, "[2000.01.02T03:04:05,2000.01.02T04:04:05>")
        s2 = self.utc.to_string(p)
        self.assertEqual(s2, "[2000.01.02T03:04:05,2000.01.02T04:04:05>")

    def test_UtcPeriod_str(self):
        c1 = api.YMDhms(2000, 01, 02, 03, 04, 05)
        t = self.utc.time(c1)
        p = api.UtcPeriod(t, t + api.deltahours(1))
        s = str(p)
        self.assertEqual(s, "[2000.01.02T03:04:05,2000.01.02T04:04:05>")


if __name__ == "__main__":
    unittest.main()

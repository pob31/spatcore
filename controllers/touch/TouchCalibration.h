#pragma once

/**
    Maps a touch panel position to a position on the display it covers, both
    as fractions (0..1) - of the panel's logical range on one side, of the
    display's bounds on the other.

    Fractions, not pixels, because the user can change the display's scaling
    at any time (the 16" ILITEK panel defaults to 1920x1200 points on a
    3840x2400 native mode): a calibration made at one resolution stays right
    at every other, and the caller turns the display fraction into points with
    the display's current bounds.

    The map is affine (display = A * panel + b), which covers offset, scale,
    rotation, mirroring and a small shear: everything a mounted panel does,
    short of a lens. The default is the identity, the panel's full range on
    the display's full bounds. fit() takes three or more measured pairs (the
    four corners and the centre, say) and solves them by least squares.

    Header-only and dependency-free.
*/

#include <cmath>
#include <cstddef>

namespace spatcore::controllers {

struct TouchCalibration
{
    // display.x = a*panel.x + b*panel.y + c;  display.y = d*panel.x + e*panel.y + f
    double a = 1, b = 0, c = 0;
    double d = 0, e = 1, f = 0;

    bool isIdentity() const noexcept
    {
        return a == 1 && b == 0 && c == 0 && d == 0 && e == 1 && f == 0;
    }

    void apply (float panelX, float panelY, float& displayX, float& displayY) const noexcept
    {
        displayX = (float) (a * panelX + b * panelY + c);
        displayY = (float) (d * panelX + e * panelY + f);
    }

    /** Least-squares fit from `count` pairs (panel[i] -> display[i], each an
        {x, y} pair of fractions). Returns false and leaves the calibration
        unchanged when there are fewer than three pairs or they are collinear. */
    bool fit (const float (*panel)[2], const float (*display)[2], size_t count)
    {
        if (count < 3)
            return false;

        // Normal equations for [x y 1] * [a b c]^T = X, shared by both rows.
        double sxx = 0, sxy = 0, sx = 0, syy = 0, sy = 0, n = (double) count;
        double tX[3] = {}, tY[3] = {};
        for (size_t i = 0; i < count; ++i)
        {
            const double x = panel[i][0], y = panel[i][1];
            sxx += x * x; sxy += x * y; sx += x; syy += y * y; sy += y;
            tX[0] += x * display[i][0]; tX[1] += y * display[i][0]; tX[2] += display[i][0];
            tY[0] += x * display[i][1]; tY[1] += y * display[i][1]; tY[2] += display[i][1];
        }

        const double m[3][3] = { { sxx, sxy, sx }, { sxy, syy, sy }, { sx, sy, n } };
        const double det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                         - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                         + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        if (std::abs (det) < 1e-12)
            return false;

        auto solve = [&] (const double (&t)[3], double& p, double& q, double& r)
        {
            // Cramer's rule: replace one column of m with t at a time.
            auto det3 = [] (const double (&k)[3][3])
            {
                return k[0][0] * (k[1][1] * k[2][2] - k[1][2] * k[2][1])
                     - k[0][1] * (k[1][0] * k[2][2] - k[1][2] * k[2][0])
                     + k[0][2] * (k[1][0] * k[2][1] - k[1][1] * k[2][0]);
            };
            double k[3][3];
            for (int col = 0; col < 3; ++col)
            {
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        k[i][j] = j == col ? t[i] : m[i][j];
                (col == 0 ? p : col == 1 ? q : r) = det3 (k) / det;
            }
        };

        solve (tX, a, b, c);
        solve (tY, d, e, f);
        return true;
    }
};

} // namespace spatcore::controllers

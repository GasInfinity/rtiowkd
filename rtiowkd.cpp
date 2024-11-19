#include "rnm/vec.hpp"
#include <rnm/rnm.hpp>
#include <rnm/format.hpp>

#include <iostream>
#include <fstream>
#include <cstdint>
#include <optional>
#include <vector>
#include <cmath>

typedef std::uint8_t u8;
typedef std::uint16_t u16;
typedef std::size_t usize;
typedef float f32;
typedef rnm::vec3<float> vec3f;
typedef vec3f color;

struct ray {
    constexpr ray() {}
    constexpr ray(const vec3f& origin, const vec3f& direction) : origin(origin), direction(direction) {}

    constexpr vec3f org() const { return origin; }
    constexpr vec3f dir() const { return direction; }
    constexpr vec3f at(f32 t) const { return origin + direction * t; }
private:
    vec3f origin;
    vec3f direction;
};

struct sphere {
    constexpr sphere() {}
    constexpr sphere(const vec3f& center, f32 radius) : center(center), radius(radius) {}

    vec3f cent() const { return center; }
    f32 rad() const { return radius; }

    // x^2 + y^2 + z^2 = r^2
    // (C - P)*(C - P) = r^2
    // (C - O - Dt)*(C - O - Dt) = r^2
    // (-Dt + (C - O))*(-Dt + (C - O)) = r^2
    // d*d*t^2 - 2*d*(C- O)t + |(C - O)|^2 = r^2 
    // |d|^2 * t^2 - t*2*(d(C - O)) + |(C - O)|^2 - r^2 = 0
    // quadratic equation
    std::optional<f32> tintersection(const ray& r) const {
        vec3f pos_diff = (center - r.org());
        f32 a = rnm::length_sqr(r.dir());
        f32 h = rnm::dot(r.dir(), pos_diff); 
        f32 c = rnm::length_sqr(pos_diff) - radius * radius;
        f32 term_sqr = h*h - a*c;
        
        if(term_sqr < 0) {
            return std::nullopt;
        } 

        f32 min_solution = (h - std::sqrt(term_sqr)) / a;
        return min_solution < 0 ? std::nullopt : std::make_optional(min_solution);
    }
private:
    vec3f center;
    f32 radius;
};

constexpr usize image_width = 800;
constexpr usize image_height = 600;
constexpr f32 aspect_ratio = static_cast<f32>(image_width)/image_height;

constexpr f32 viewport_height = 2.0;
constexpr f32 viewport_width = viewport_height * aspect_ratio;

constexpr f32 camera_focal_length = 1.0;
constexpr vec3f camera_center = vec3f(0, 0, 0);
constexpr vec3f viewport_u = vec3f(viewport_width, 0, 0);
constexpr vec3f viewport_v = vec3f(0, -viewport_height, 0);
constexpr vec3f pixel_delta_u = viewport_u / f32(image_width);
constexpr vec3f pixel_delta_v = viewport_v / f32(image_height);
constexpr vec3f viewport_upper_left = camera_center - vec3f(0, 0, camera_focal_length) - viewport_u / f32(2) - viewport_v / f32(2);
constexpr vec3f p00 = viewport_upper_left + .5f * (pixel_delta_u + pixel_delta_v); 

void renderPPM(std::ostream& output, usize w, usize h, const color* data) {
    output << "P3\n" << w << ' ' << h << "\n255\n";
    
    for (usize i = 0; i < h; ++i) {
        for (usize j = 0; j < w; ++j) {
            const color& p = data[i*w+j]; 
            u8 r = p.r * 255.0;
            u8 g = p.g * 255.0;
            u8 b = p.b * 255.0;
            output << static_cast<u16>(r) << ' ' << static_cast<u16>(g) << ' ' << static_cast<u16>(b) << '\n';
        }
    }
}

constexpr sphere s = sphere(vec3f{0, 0, -1}, .5f);

color ray_color(const ray& ray) {
    std::optional<f32> tinter = s.tintersection(ray);
    if(tinter) {
        vec3f intersection = ray.at(*tinter);
        vec3f normal = rnm::normalized(intersection - s.cent());
        return (normal+color(1.0f))*.5f;
    }

    return color(0, 0, 0);
}

int main() {
    std::vector<color> image;
    image.reserve(image_width*image_height);
    image.resize(image_width*image_height);

    for (usize i = 0; i < image_height; ++i) {
        if(i % 50) {
            std::clog << std::format("Progress... {:.1f}%\n", (i / static_cast<f32>(image_height)) * 100);
        }

        for (usize j = 0; j < image_width; ++j) {
            const vec3f pixel_position = p00 + pixel_delta_u * f32(j) + pixel_delta_v * f32(i);
            const vec3f ray_direction = pixel_position - camera_center;
            const ray ray{camera_center, ray_direction};
            image[i*image_width+j] = ray_color(ray);
        }
    } 

    std::ofstream output("out.ppm");
    renderPPM(output, image_width, image_height, image.data());
    return 0;
}

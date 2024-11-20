#include <random>
#include <rnm/rnm.hpp>
#include <rnm/format.hpp>

#include <iostream>
#include <fstream>
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>
#include <cmath>

typedef std::uint8_t u8;
typedef std::uint16_t u16;
typedef std::size_t usize;
typedef float f32;
typedef rnm::vec2<float> vec2f;
typedef rnm::vec3<float> vec3f;
typedef vec3f color;


struct ray {
    constexpr ray(const vec3f& origin, const vec3f& direction) : origin(origin), direction(direction) {}

    constexpr vec3f org() const { return origin; }
    constexpr vec3f dir() const { return direction; }
    constexpr vec3f at(f32 t) const { return origin + direction * t; }
private:
    vec3f origin;
    vec3f direction;
};

struct ray_hit {
    constexpr ray_hit(f32 at, const vec3f& point, const vec3f& normal) : at(at), point(point), normal(normal) {}
    
    constexpr f32 t() const { return at; }
    constexpr vec3f pt() const { return point; }
    constexpr vec3f norm() const { return normal; }
private:
    f32 at;
    vec3f point;
    vec3f normal;
};

struct sphere {
    constexpr sphere(const vec3f& center, f32 radius) : center(center), radius(radius) {}

    constexpr vec3f cent() const { return center; }
    constexpr f32 rad() const { return radius; }

    // x^2 + y^2 + z^2 = r^2
    // (C - P)*(C - P) = r^2
    // (C - O - Dt)*(C - O - Dt) = r^2
    // (-Dt + (C - O))*(-Dt + (C - O)) = r^2
    // d*d*t^2 - 2*d*(C- O)t + |(C - O)|^2 = r^2 
    // |d|^2 * t^2 - t*2*(d(C - O)) + |(C - O)|^2 - r^2 = 0
    // quadratic equation
    constexpr std::optional<ray_hit> intersection(const ray& r, f32 min, f32 max) const {
        vec3f pos_diff = (center - r.org());
        f32 a = rnm::length_sqr(r.dir());
        f32 h = rnm::dot(r.dir(), pos_diff); 
        f32 c = rnm::length_sqr(pos_diff) - radius * radius;
        f32 term_sqr = h*h - a*c;
        
        if(term_sqr < 0) {
            return std::nullopt;
        } 

        f32 term = std::sqrt(term_sqr);
        f32 first_solution = (h - term) / a;
        
        f32 t = first_solution;
        if(first_solution < min || first_solution > max) {
            f32 second_solution = (h + term) / a;

            if(second_solution < min || second_solution > max) {
                return std::nullopt;
            }

            t = second_solution;
        }
        
        vec3f point = r.at(t);
        vec3f normal = (point - center) / radius;
        return ray_hit{t, point, normal}; 
    }
private:
    vec3f center;
    f32 radius;
};

struct hittable {
    std::variant<sphere> obj;

    constexpr hittable(const sphere& sphere) : obj(sphere) {}

    constexpr std::optional<ray_hit> intersection(const ray& r, f32 min, f32 max) const {
        return std::visit([&r, &min, &max](auto&& v) { return v.intersection(r, min, max); }, this->obj);
    }
};

class universe {
public:
    universe& add(hittable&& hitt) {
        this->objects.push_back(std::move(hitt));
        return *this;
    }

    std::optional<ray_hit> intersection(const ray& r, f32 tmin, f32 tmax) const {
        std::optional<ray_hit> closest_hit = std::nullopt;
        f32 closest = tmax;

        for (const hittable& hitt : this->objects) {
            std::optional<ray_hit> hit = hitt.intersection(r, tmin, closest);

            if(hit) {
                closest = hit->t();
                closest_hit = hit;
            }
        }

        return closest_hit;
    }
private:
    std::vector<hittable> objects;
};

struct camera {
    camera(const vec3f& position, f32 focal_length, f32 viewport_height, f32 samples_per_pixel)
        : position(position), focal_length(focal_length), viewport_height(viewport_height), samples_per_pixel(samples_per_pixel), sampling_entropy(), sampling_distribution(0, 1) { }

    void render(const universe& environ, usize width, usize height, std::span<color> data) {
        const f32 aspect = width / static_cast<f32>(height);
        const f32 viewport_width = viewport_height * aspect;

        const vec3f viewport_u = vec3f(viewport_width, 0, 0);
        const vec3f viewport_v = vec3f(0, -viewport_height, 0);
        const vec3f pixel_delta_u = viewport_u / width;
        const vec3f pixel_delta_v = viewport_v / height;
        const vec3f viewport_upper_left = position - vec3f(0, 0, focal_length) - viewport_u / 2 - viewport_v / 2;
        const vec3f p00 = viewport_upper_left + .5f * (pixel_delta_u + pixel_delta_v); 
        
        const f32 samples_scale = 1.f / samples_per_pixel;

        for (usize i = 0; i < height; ++i) {
            for (usize j = 0; j < width; ++j) {
                color pixel_samples = {};

                for (usize k = 0; k < samples_per_pixel; ++k) {
                    const vec2f random_sample = random_sample_vec2f();
                    const vec3f pixel_position = p00 + pixel_delta_u * (j + random_sample.x) + pixel_delta_v * (i + random_sample.y);

                    const vec3f ray_direction = pixel_position - position;
                    const ray ray{position, ray_direction};

                    pixel_samples += ray_color(environ, ray);
                }

                data[i*width+j] = pixel_samples * samples_scale;
            }
        }
    }
private:
    color ray_color(const universe& environ, const ray& ray) {
        std::optional<ray_hit> hit = environ.intersection(ray, 0, std::numeric_limits<f32>::infinity());
        if(hit) {
            return (hit->norm()+color(1.0f))*.5f;
        }

        return color(0, 0, 0);
    }
    
    vec2f random_sample_vec2f() {
        return {sampling_distribution(sampling_entropy) - .5f, sampling_distribution(sampling_entropy) - .5f}; 
    }

    vec3f position;
    f32 focal_length;
    f32 viewport_height;
    usize samples_per_pixel;
    std::default_random_engine sampling_entropy;
    std::uniform_real_distribution<f32> sampling_distribution;
};

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


constexpr usize image_width = 800;
constexpr usize image_height = 600;

int main() {
    std::vector<color> image;
    image.reserve(image_width*image_height);
    image.resize(image_width*image_height);

    const universe environ = universe()
    .add(sphere{{0, 0, -1}, .5f})
    .add(sphere{{0, -100.5, -1}, 100});

    camera cam{{0, 0, 0}, 1.f, 2.f, 16};
    cam.render(environ, image_width, image_height, image);

    std::ofstream output("out.ppm");
    renderPPM(output, image_width, image_height, image.data());
    return 0;
}

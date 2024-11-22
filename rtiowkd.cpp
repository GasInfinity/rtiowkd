#include <rnm/rnm.hpp>
#include <rnm/format.hpp>

#include <iostream>
#include <fstream>
#include <cstdint>
#include <optional>
#include <utility>
#include <variant>
#include <vector>
#include <algorithm>
#include <random>
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

struct material;

struct ray_hit {
    constexpr ray_hit(f32 at, const vec3f& point, const vec3f& normal, const material& material) : at(at), point(point), normal(normal), material(material) {}
    
    constexpr f32 t() const { return at; }
    constexpr vec3f pt() const { return point; }
    constexpr vec3f norm() const { return normal; }
    constexpr const material& mat() const { return material; }
private:
    f32 at;
    vec3f point;
    vec3f normal;
    const material& material;
};

std::default_random_engine default_random = std::default_random_engine{std::random_device{}()};
std::uniform_real_distribution<f32> default_real_distribution(0.f, 1.f);

template<std::uniform_random_bit_generator Gen>
vec3f random_vec3f(Gen& random, f32 min, f32 max) {
    std::uniform_real_distribution<f32> distribution(min, max);
    return {distribution(random), distribution(random), distribution(random)};
}

template<std::uniform_random_bit_generator Gen>
vec3f random_unit_vec3f(Gen& random) {
    vec3f v = random_vec3f(random, -1, 1);

    while (std::abs(rnm::length_sqr(v)) < .0000001f) {
        v = random_vec3f(random, -1, 1);
    }

    return rnm::normalized(v);
}

struct lambertian_material {
    color albedo;
    f32 reflectance;

    constexpr lambertian_material(color albedo, f32 reflectance = 0.9f) : albedo(albedo), reflectance(reflectance) { }

    std::pair<color, std::optional<vec3f>> scatter(const ray& r, const ray_hit& hit) const {
        if (default_real_distribution(default_random) > reflectance) {
            return std::make_pair(albedo, std::nullopt);
        }

        vec3f reflect_direction = random_unit_vec3f(default_random);

        if(rnm::dot(reflect_direction, hit.norm()) < 0)
            reflect_direction = -reflect_direction;

        return std::make_pair(albedo, rnm::normalized(hit.norm() + reflect_direction));
    }
};

struct metal_material {
    color albedo;
    f32 fuzz_factor;

    constexpr metal_material(color albedo, f32 fuzz_factor = 1.f) : albedo(albedo), fuzz_factor(fuzz_factor) { }

    std::pair<color, std::optional<vec3f>> scatter(const ray& r, const ray_hit& hit) const {
        vec3f reflect_direction = rnm::reflect(r.dir(), hit.norm());
        vec3f scatter_direction = rnm::normalized(reflect_direction + random_unit_vec3f(default_random) * fuzz_factor);

        if(rnm::dot(scatter_direction, hit.norm()) <= 0) {
            return std::make_pair(albedo, std::nullopt);
        }
        
        return std::make_pair(albedo, scatter_direction);
    }
};

struct dielectric_material {
    f32 refraction_index;

    constexpr dielectric_material(f32 refraction_index) : refraction_index(refraction_index) { }

    std::pair<color, std::optional<vec3f>> scatter(const ray& r, const ray_hit& hit) const {
        bool outward = rnm::dot(r.dir(), hit.norm()) < 0;
        f32 refraction = outward ? 1.f / this->refraction_index : this->refraction_index;
        vec3f refract_normal = (outward * 2.f - 1.f) * hit.norm();
        
        f32 cos_theta = std::min(rnm::dot(-r.dir(), refract_normal), 1.f);
        f32 sin_theta = std::sqrt(1 - cos_theta * cos_theta);

        if(refraction * sin_theta > 1.f || reflectance(cos_theta, refraction) > default_real_distribution(default_random)) {
            return std::make_pair(color(1.f), rnm::reflect(r.dir(), hit.norm()));
        }

        vec3f refract_direction = rnm::refract(r.dir(), refract_normal, refraction);

        return std::make_pair(color(1.f), refract_direction);
    }

private:
    constexpr static double reflectance(f32 cos, f32 refraction) {
        f32 r0 = (1 - refraction) / (1 + refraction);
        f32 r1 = r0 * r0;
        f32 oneMinusCos = 1 - cos;
        f32 oneMinusCos2 = oneMinusCos*oneMinusCos;
        return r1 + (1 - r0) * oneMinusCos2 * oneMinusCos2 * oneMinusCos;
    }
};

struct material {
    std::variant<lambertian_material, metal_material, dielectric_material> obj;

    constexpr std::pair<color, std::optional<vec3f>> scatter(const ray& r, const ray_hit& hit) const {
        return std::visit([&r, &hit](auto&& v) { return v.scatter(r, hit); }, this->obj);
    }
};

struct sphere {
    constexpr sphere(const vec3f& center, f32 radius, const material& mat) : center(center), radius(radius), mat(mat) {}

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
        return ray_hit{t, point, normal, mat}; 
    }
private:
    vec3f center;
    f32 radius;
    material mat;
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
                closest_hit.emplace(hit->t(), hit->pt(), hit->norm(), hit->mat());
            }
        }

        return closest_hit;
    }
private:
    std::vector<hittable> objects;
};

struct camera {
    camera(const vec3f& position, f32 focal_length, f32 viewport_height, f32 samples_per_pixel, f32 ray_depth)
        : position(position), focal_length(focal_length), viewport_height(viewport_height), samples_per_pixel(samples_per_pixel), ray_depth(ray_depth), entropy(std::random_device{}()), sampling_distribution(-.5f, .5f) { }

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

                    const vec3f ray_direction = rnm::normalized(pixel_position - position);
                    const ray ray{position, ray_direction};

                    pixel_samples += ray_color(environ, ray, ray_depth);
                }

                data[i*width+j] = pixel_samples * samples_scale;
            }
        }
    }
private:
    color ray_color(const universe& environ, const ray& ray, usize depth) {
        if(depth == 0) {
            return color{0, 0, 0};
        }

        std::optional<ray_hit> hit = environ.intersection(ray, 0.001f, std::numeric_limits<f32>::infinity());

        if(hit) {
            const material& mat = hit->mat();
            std::pair<color, std::optional<vec3f>> scatter = mat.scatter(ray, *hit);

            if(scatter.second) {
                return rnm::mul(scatter.first, ray_color(environ, {hit->pt(), *scatter.second}, depth - 1));
            }

            return scatter.first;
        }

        return rnm::lerp(color(1.f), color(.5, .7, 1.f), (ray.dir().y + 1.f) * .5f);
    }
    
    vec2f random_sample_vec2f() {
        return {sampling_distribution(entropy), sampling_distribution(entropy)}; 
    }


    vec3f position;
    f32 focal_length;
    f32 viewport_height;
    usize samples_per_pixel;
    usize ray_depth;
    std::default_random_engine entropy;
    std::uniform_real_distribution<f32> sampling_distribution;
};

inline color gamma_correct(const color& c) {
    f32 x = c.x > 0.f ? std::sqrt(c.x) : c.x;
    f32 y = c.y > 0.f ? std::sqrt(c.y) : c.y;
    f32 z = c.z > 0.f ? std::sqrt(c.z) : c.z;
    return {x, y, z};
}

void renderPPM(std::ostream& output, usize w, usize h, const color* data) {
    output << "P3\n" << w << ' ' << h << "\n255\n";
    
    for (usize i = 0; i < h; ++i) {
        for (usize j = 0; j < w; ++j) {
            const color& p = gamma_correct(data[i*w+j]); 
            u8 r = std::clamp<f32>(p.r * 255.0, 0.f, 255.f);
            u8 g = std::clamp<f32>(p.g * 255.0, 0.f, 255.f);
            u8 b = std::clamp<f32>(p.b * 255.0, 0.f, 255.f);
            output << static_cast<u16>(r) << ' ' << static_cast<u16>(g) << ' ' << static_cast<u16>(b) << '\n';
        }
    }
}


constexpr usize image_width = 400;
constexpr usize image_height = 225;

int main() {
    std::vector<color> image;
    image.reserve(image_width*image_height);
    image.resize(image_width*image_height);

    const universe environ = universe()
    .add(sphere{{0, -100.5, -1}, 100, material{lambertian_material{{0.8, 0.8, 0.0}}}})
    .add(sphere{{0, 0, -1.2f}, .5f, material{lambertian_material{{0.1, 0.2, 0.5}}}})
    .add(sphere{{-1, 0, -1}, .5f, material{dielectric_material{1.5f}}})
    .add(sphere{{-1, 0, -1}, .4f, material{dielectric_material{1.f / 1.5f}}})
    .add(sphere{{1, 0, -1}, .5f, material{metal_material{{.8f, .6f, .2f}, 1.f}}});

    camera cam{{0, 0, 0}, 1.f, 2.f, 320, 50};
    cam.render(environ, image_width, image_height, image);

    std::ofstream output("out.ppm");
    renderPPM(output, image_width, image_height, image.data());
    return 0;
}

/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#define _USE_MATH_DEFINES

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    // Set number of particles
    num_particles = 25;

    // Create normal distributions for x, y and theta initial poses
    default_random_engine gen;
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    // Create initial particles
    for (int i = 0; i < num_particles; i++)
    {
        // Create particle
        Particle particle;
        particle.id = i;
        particle.x = dist_x(gen);
        particle.y = dist_y(gen);
        particle.theta = dist_theta(gen);
        particle.weight = 1;

        // Add to list
        particles.push_back(particle);
        weights.push_back(particle.weight);
    }

    // Set initialisation flag to true
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    // Create noise model
    default_random_engine gen;
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);

    for (int i = 0; i < num_particles; i++)
    {
        // Update position based on model
        Particle & particle = particles[i];
        if (abs(yaw_rate) > 0)
        {
            particle.x += (velocity / yaw_rate) * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
            particle.y += (velocity / yaw_rate) * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));
            particle.theta += yaw_rate * delta_t;
        }
        else
        {
            particle.x += velocity * delta_t * sin(particle.theta);
            particle.y += velocity * delta_t * cos(particle.theta);
        }

        // Add noise

        particle.x += dist_x(gen);
        particle.y += dist_y(gen);
        particle.theta += dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations, double sensor_range) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

    for (int i = 0; i < observations.size(); i++)
    {
        LandmarkObs & measurement = observations[i];

        measurement.id = 0;
        double bestAssociationDistance = sensor_range;
        for (int j = 0; j < predicted.size(); j++)
        {
            const LandmarkObs & prediction = predicted[j];

            double distance = dist(measurement.x, measurement.y, prediction.x, prediction.y);
            if (distance < bestAssociationDistance)
            {
                measurement.id = prediction.id;
                bestAssociationDistance = distance;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    for (int i = 0; i < num_particles; i++)
    {
        Particle & particle = particles[i];

        // Calculate predicted measurement to each feature
        std::vector<LandmarkObs> predictions;
        for (int k = 0; k < map_landmarks.landmark_list.size(); k++)
        {
            const Map::single_landmark_s & landmark = map_landmarks.landmark_list[k];

            // Calculate offset to feature from particle
            double dx = landmark.x_f - particle.x;
            double dy = landmark.y_f - particle.y;

            // Convert into vehicle frame
            double sense_x = dx * cos(particle.theta) + dy * sin(particle.theta);
            double sense_y = -dx * sin(particle.theta) + dy * cos(particle.theta);

            // Create prediction
            LandmarkObs predicted;
            predicted.id = landmark.id_i;
            predicted.x = sense_x;
            predicted.y = sense_y;

            // Add to list
            predictions.push_back(predicted);
        }

        // Associate measurements based on predicted measurements
        dataAssociation(predictions, observations, sensor_range);

        // Calculate new particle weight and associations
        std::vector<int> associations;
        std::vector<double> sense_x;
        std::vector<double> sense_y;

        double weight = 1;
        for (int j = 0; j < observations.size(); j++)
        {
            const LandmarkObs & measurement = observations[j];

            for (int k = 0; k < predictions.size(); k++)
            {
                const LandmarkObs & prediction = predictions[k];

                if (prediction.id == measurement.id)
                {
                    double dx = measurement.x - prediction.x;
                    double dy = measurement.y - prediction.y;
                    double w_j = exp(-0.5 * (dx * dx / (std_landmark[0] * std_landmark[0]) + dy * dy / (std_landmark[1] * std_landmark[1]))) /
                        sqrt(2 * M_PI * std_landmark[0] * std_landmark[1]);

                    weight *= w_j;

                    associations.push_back(measurement.id);
                    sense_x.push_back(particle.x + measurement.x * cos(-particle.theta) + measurement.y * sin(-particle.theta));
                    sense_y.push_back(particle.y - measurement.x * sin(-particle.theta) + measurement.y * cos(-particle.theta));

                    break;
                }
            }
        }

        particle.weight = weight;
        weights[i] = particle.weight;

        particle = SetAssociations(particle, associations, sense_x, sense_y);
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    // Create distribution that returns particle indices with probability based on weight of particle
    default_random_engine gen;
    discrete_distribution<int> distribution(weights.begin(), weights.end());

    std::vector<Particle> new_particles;

    for (int i = 0; i < num_particles; i++)
    {
        new_particles.push_back(particles[distribution(gen)]);
    }

    particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
